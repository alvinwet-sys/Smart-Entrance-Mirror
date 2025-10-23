#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语音唤醒检测器
- 使用 MFCC + DTW 进行关键词检测
- 使用 WebRTC VAD 进行语音活动检测
- 集成到事件总线系统中
"""

import asyncio
import logging
import queue
import uuid
import os
import sys
import json
import time
from datetime import datetime
from typing import Callable, Awaitable, Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

try:
    import numpy as np
    import sounddevice as sd
    import webrtcvad
    import librosa
except ImportError as e:
    print(f"缺少必要的库: {e}", file=sys.stderr)
    print("请运行: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# 从工程目录导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_llm.wakeup_core import DTWKWS, VADHelper


# 配置日志
logger = logging.getLogger(__name__)

import numpy as np
import sounddevice as sd
import webrtcvad
import os
import librosa
import traceback

# =========================
# 配置（只改这里就行）
# =========================
CONFIG = {
    # 模板 .wav 路径（可 1~N 个；建议 1~3 条同词的不同说法）
    "keywords": [
        "./templates/kw1.wav",
        # "./templates/kw2.wav",
        # "./templates/kw3.wav",
    ],
    # 每个模板对应的显示名（可与 keywords 同长；不填则自动用文件名）
    "keyword_names": [
        "你好小镜",
        # "你好小镜",
        # "你好小镜",
    ],
    # DTW 距离阈值（越小越严格；常见 200~320 之间现场调）
    "dtw_threshold": 250.0,

    # 事件中的通道标记
    "channel": "near",  # "near" | "far"

    # 采样与帧
    "sample_rate": 16000,
    "frame_length": 512,           # ≈32ms @16k，与原切帧逻辑一致
    "max_window_sec": 1.0,         # DTW 滑窗最大长度
    "min_window_sec": 0.35,        # DTW 滑窗最小长度（不足不比）
    "cooldown_sec": 0.8,           # 触发后冷却时间

    # VAD 配置
    "vad_aggressiveness": 2,       # 0~3（越大越苛刻）
    "vad_majority_frames": 10,     # 多数表决窗口帧数
    "vad_frame_ms": 20,            # 每帧时长（ms）

    # 声卡设备（None=默认输入设备；或设置为设备ID整数）
    "device": None,

    # 日志配置（只影响 stderr，不影响 stdout JSON）
    "log_enable_info": True,
    "log_enable_warn": True,
    "log_enable_error": True,
    "log_enable_debug": False,     # 调试多时可打开 True
    "log_with_time": True,         # 打印时间戳
}


# =========================
# 简易日志（统一走 stderr）
# =========================
def _now_str() -> str:
    if not CONFIG.get("log_with_time", True):
        return ""
    # 采用本地时间；如需UTC可改 time.gmtime()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _emit_log(level: str, msg: str):
    ts = _now_str()
    prefix = f"[{level}] " if level else ""
    line = f"{prefix}{msg}" if not ts else f"{ts} {prefix}{msg}"
    # 只走 stderr，避免污染 stdout 的 JSON 日志
    print(line, file=sys.stderr)

def log_info(msg: str):
    if CONFIG.get("log_enable_info", True):
        _emit_log("INFO", msg)

def log_warn(msg: str):
    if CONFIG.get("log_enable_warn", True):
        _emit_log("WARN", msg)

def log_error(msg: str):
    if CONFIG.get("log_enable_error", True):
        _emit_log("ERROR", msg)

def log_debug(msg: str):
    if CONFIG.get("log_enable_debug", False):
        _emit_log("DEBUG", msg)


# =========================
# 工具
# =========================
def now_ts() -> float:
    return time.time()

def gen_trace_id() -> str:
    return uuid.uuid4().hex

def int16_from_float32(float32_pcm: np.ndarray) -> np.ndarray:
    float32_pcm = np.clip(float32_pcm, -1.0, 1.0)
    return (float32_pcm * 32767).astype(np.int16)


# =========================
# 事件输出（stdout）
# =========================
class WakeupEmitter:
    """
    将唤醒事件按 Wakeup Schema 输出到 stdout。
    """
    def __init__(self, channel: str = "near"):
        self.channel = channel

    def emit(self, keyword_name: str, confidence: float, vad_flag: bool):
        event = {
            "ts": now_ts(),
            "trace_id": gen_trace_id(),
            "source": "voice",
            "keyword": keyword_name,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "channel": self.channel,
            "vad": bool(vad_flag),
        }
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


class ErrorEmitter:
    """
    统一错误事件输出（你的 ErrorEvent Schema）
    required: ts, trace_id, source, error_code, message
    optional: retry_after_ms
    """
    def __init__(self, source: str = "wakeup"):
        self.source = source

    def emit(self, error_code: str, message: str, retry_after_ms: int | None = None):
        event = {
            "ts": now_ts(),
            "trace_id": gen_trace_id(),
            "source": self.source,
            "error_code": error_code,
            "message": str(message),
        }
        if retry_after_ms is not None:
            event["retry_after_ms"] = max(0, int(retry_after_ms))
        # 事件走 stdout（结构化）
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


# =========================
# VAD 多数表决
# =========================
class VADHelper:
    """
    WebRTC VAD：用于给事件加 'vad' 标记；多数表决稳定输出。
    """
    def __init__(self, sample_rate: int, aggressiveness: int = 2, majority_frames: int = 10, frame_ms: int = 20):
        import collections
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit PCM
        self.buf = collections.deque(maxlen=majority_frames)

    def is_speech(self, pcm16: bytes) -> bool:
        vad_frames = [pcm16[i:i + self.frame_bytes] for i in range(0, len(pcm16), self.frame_bytes)]
        for f in vad_frames:
            if len(f) == self.frame_bytes:
                self.buf.append(1 if self.vad.is_speech(f, self.sample_rate) else 0)
        if len(self.buf) == 0:
            return False
        return sum(self.buf) >= (len(self.buf) // 2 + 1)


# =========================
# MFCC + DTW
# =========================
def _mfcc_feature(y: np.ndarray, sr: int) -> np.ndarray:
    # 20维 MFCC，10ms hop，40个 mel 滤波器
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=int(0.01*sr), n_mels=40
    )

def _dtw_distance(A: np.ndarray, B: np.ndarray) -> float:
    # A/B: (n_mfcc, T)；返回终点累积距离
    D, _ = librosa.sequence.dtw(X=A, Y=B, metric="euclidean")
    return float(D[-1, -1])

class DTWKWS:
    """
    模板 .wav + MFCC + DTW 的 KWS 实现（接口尽量贴近 PorcupineKWS）
    - sample_rate: 16000
    - frame_length: 512（≈32ms）
    - process(frame:int16[frame_length]) -> 命中模板索引或 -1
    """
    def __init__(self, keyword_paths: List[str], dtw_threshold: float,
                 sample_rate: int, frame_length: int,
                 max_window_sec: float, min_window_sec: float, cooldown_sec: float):
        if len(keyword_paths) == 0:
            raise ValueError("KEYWORD_MISSING: 请至少提供一个 模板 .wav 路径")

        self.sample_rate = int(sample_rate)
        self.frame_length = int(frame_length)
        self.dtw_threshold = float(dtw_threshold)

        # 载入模板并预先计算特征
        self.templates = []
        self.template_names = []
        for p in keyword_paths:
            if not os.path.exists(p):
                raise ValueError(f"INVALID_ARGUMENT: 模板不存在 {p}")
            y, sr = librosa.load(p, sr=self.sample_rate, mono=True)
            if y.size == 0:
                raise ValueError(f"INVALID_ARGUMENT: 模板为空 {p}")
            F = _mfcc_feature(y, self.sample_rate)
            self.templates.append(F)
            self.template_names.append(os.path.splitext(os.path.basename(p))[0])

        # 滑窗缓冲
        self.max_window_samples = int(max_window_sec * self.sample_rate)
        self.min_window_samples = int(min_window_sec * self.sample_rate)
        self.buffer = np.zeros((0,), dtype=np.int16)

        # 触发冷却
        self.cooldown_samples = int(cooldown_sec * self.sample_rate)
        self.cooldown_left = 0

        # 上次匹配信息
        self.last_match_confidence = 0.0
        self.last_match_index = -1

    def process(self, pcm16_frame: np.ndarray) -> int:
        """
        输入：int16 一帧
        输出：命中模板索引（>=0）或 -1
        """
        if pcm16_frame.dtype != np.int16:
            pcm16_frame = np.asarray(pcm16_frame, dtype=np.int16)

        # 冷却期间：更新缓冲但不检测
        if self.cooldown_left > 0:
            self.cooldown_left -= len(pcm16_frame)
            self.buffer = np.concatenate([self.buffer, pcm16_frame])[-self.max_window_samples:]
            return -1

        # 更新缓冲
        self.buffer = np.concatenate([self.buffer, pcm16_frame])[-self.max_window_samples:]

        # 缓冲长度不足
        if len(self.buffer) < self.min_window_samples:
            return -1

        # 取最近窗口（最多 ~0.9s）
        win = self.buffer[-min(self.max_window_samples, int(0.9 * self.sample_rate)):]
        y = (win.astype(np.float32) / 32768.0)
        feat_cand = _mfcc_feature(y, self.sample_rate)

        # 与模板 DTW 匹配
        dmin = 1e9
        imin = -1
        for i, F in enumerate(self.templates):
            d = _dtw_distance(F, feat_cand)
            if d < dmin:
                dmin, imin = d, i

        if dmin <= self.dtw_threshold:
            self.cooldown_left = self.cooldown_samples
            self.last_match_confidence = float(np.clip(1.0 - dmin / (self.dtw_threshold + 1e-6), 0.0, 1.0))
            self.last_match_index = imin
            return imin

        return -1

    def get_last_confidence(self) -> float:
        return float(self.last_match_confidence)


logger = logging.getLogger(__name__)

class WakeupDetector:
    """语音唤醒检测器"""
    
    def __init__(self, config: Dict[str, Any], publish: Callable[[str, dict], Awaitable[bool]]):
        """初始化唤醒检测器
        
        Args:
            config: 配置信息
            publish: 事件发布函数
        """
        self.config = config
        self.publish = publish
        self.running = False
        self._audio_queue = queue.Queue()
        
        # 初始化组件
        self._setup_components()
        
    def _setup_components(self):
        """初始化检测组件"""
        try:
            # 检查模板文件是否存在
            for keyword_path in self.config["keywords"]:
                if not os.path.exists(keyword_path):
                    raise FileNotFoundError(f"模板文件不存在: {keyword_path}")
            
            # 初始化关键词检测
            self.kws = DTWKWS(
                keyword_paths=self.config["keywords"],
                dtw_threshold=self.config["dtw_threshold"],
                sample_rate=self.config["sample_rate"],
                frame_length=self.config["frame_length"],
                max_window_sec=self.config["max_window_sec"],
                min_window_sec=self.config["min_window_sec"],
                cooldown_sec=self.config["cooldown_sec"]
            )
            
            # 初始化VAD
            self.vad = VADHelper(
                sample_rate=self.config["sample_rate"],
                aggressiveness=self.config["vad_aggressiveness"],
                majority_frames=self.config["vad_majority_frames"],
                frame_ms=self.config["vad_frame_ms"]
            )
            
            # 准备关键词名称
            keyword_names_cfg = self.config.get("keyword_names", [])
            if keyword_names_cfg and len(keyword_names_cfg) == len(self.config["keywords"]):
                self.keyword_names = keyword_names_cfg
            else:
                self.keyword_names = [os.path.splitext(os.path.basename(p))[0] 
                                    for p in self.config["keywords"]]
                                    
        except Exception as e:
            logger.error(f"初始化检测组件失败: {e}")
            raise
            
    async def _emit_wakeup(self, keyword_idx: int, confidence: float, vad_flag: bool):
        """发送唤醒事件"""
        try:
            # 创建唤醒事件
            event = {
                "ts": datetime.now().timestamp(),
                "trace_id": f"wakeup-{uuid.uuid4().hex[:8]}",
                "source": "voice",
                "keyword": self.keyword_names[keyword_idx],
                "confidence": float(confidence),
                "channel": self.config["channel"],
                "vad": bool(vad_flag)
            }
            
            # 通过事件总线发送
            await self.publish("voice.wakeup", event)
            logger.info(f"检测到唤醒词：{event['keyword']} (conf={confidence:.3f})")
            
        except Exception as e:
            logger.error(f"发送唤醒事件失败: {e}")
            
    async def _emit_error(self, error_code: str, message: str, retry_after_ms: int = None):
        """发送错误事件"""
        try:
            event = {
                "ts": datetime.now().timestamp(),
                "trace_id": f"wakeup-err-{uuid.uuid4().hex[:8]}",
                "source": "voice.wakeup",
                "error_code": error_code,
                "message": str(message)
            }
            if retry_after_ms is not None:
                event["retry_after_ms"] = max(0, int(retry_after_ms))
                
            await self.publish("core.error_event", event)
            
        except Exception as e:
            logger.error(f"发送错误事件失败: {e}")
            
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            logger.warning(f"音频回调状态: {status}")
        try:
            self._audio_queue.put(indata.copy(), block=False)
        except Exception as e:
            logger.debug(f"音频队列满或错误: {e}")
            
    async def _process_audio(self):
        """处理音频数据"""
        pcm_buf = bytearray()
        
        while self.running:
            try:
                # 获取音频数据
                indata = self._audio_queue.get(timeout=5.0)
                pcm16 = int16_from_float32(indata[:, 0])
                
                # VAD 检测
                pcm_buf.extend(pcm16.tobytes())
                vad_flag = self.vad.is_speech(bytes(pcm_buf))
                
                # 关键词检测
                frame_length = self.config["frame_length"]
                idx = 0
                while idx + frame_length <= len(pcm16):
                    frame = pcm16[idx:idx + frame_length]
                    idx += frame_length
                    res = self.kws.process(frame)
                    
                    if res >= 0:
                        # 检测到唤醒词
                        conf = float(self.kws.get_last_confidence())
                        await self._emit_wakeup(res, conf, vad_flag)
                        
                # 控制 VAD 缓冲区大小
                if len(pcm_buf) > self.config["sample_rate"] * 2:
                    pcm_buf = pcm_buf[-self.config["sample_rate"] * 2:]
                    
            except queue.Empty:
                await self._emit_error(
                    "AUDIO_STREAM_TIMEOUT",
                    "超过5秒未收到音频数据",
                    retry_after_ms=2000
                )
            except Exception as e:
                await self._emit_error(
                    "PROCESSING_ERROR",
                    f"音频处理异常: {e}"
                )
                
    async def start(self):
        """启动唤醒检测"""
        if self.running:
            return
            
        self.running = True
        
        try:
            # 创建音频输入流
            stream = sd.InputStream(
                samplerate=self.config["sample_rate"],
                channels=1,
                dtype="float32",
                blocksize=self.config["frame_length"],
                callback=self._audio_callback,
                device=self.config["device"]
            )
            
            logger.info("唤醒检测器已启动，等待唤醒词...")
            
            with stream:
                await self._process_audio()
                
        except Exception as e:
            await self._emit_error(
                "AUDIO_STREAM_ERROR",
                f"音频流创建失败: {e}"
            )
            self.running = False
            raise
            
    async def stop(self):
        """停止唤醒检测"""
        self.running = False
        logger.info("唤醒检测器已停止")


def main():
    """主函数"""
    # 创建错误发射器
    err = ErrorEmitter(source="wakeup")
    
    # 读配置
    keywords = CONFIG["keywords"]
    keyword_names_cfg = CONFIG.get("keyword_names") or []
    dtw_threshold = CONFIG["dtw_threshold"]
    channel = CONFIG["channel"]
    sample_rate = CONFIG["sample_rate"]
    frame_length = CONFIG["frame_length"]
    max_window_sec = CONFIG["max_window_sec"]
    min_window_sec = CONFIG["min_window_sec"]
    cooldown_sec = CONFIG["cooldown_sec"]
    vad_aggr = CONFIG["vad_aggressiveness"]
    vad_majority_frames = CONFIG["vad_majority_frames"]
    vad_frame_ms = CONFIG["vad_frame_ms"]
    device = CONFIG["device"]

    # 检查模板文件
    for keyword_path in keywords:
        if not os.path.exists(keyword_path):
            err.emit("INVALID_ARGUMENT", f"模板文件不存在: {keyword_path}")
            return

    # 初始化 KWS
    try:
        kws = DTWKWS(
            keyword_paths=keywords,
            dtw_threshold=dtw_threshold,
            sample_rate=sample_rate,
            frame_length=frame_length,
            max_window_sec=max_window_sec,
            min_window_sec=min_window_sec,
            cooldown_sec=cooldown_sec,
        )
    except ValueError as ve:
        text = str(ve)
        if text.startswith("KEYWORD_MISSING"):
            err.emit("KEYWORD_MISSING", "请至少提供一个 模板 .wav 路径")
            return
        elif text.startswith("INVALID_ARGUMENT"):
            err.emit("INVALID_ARGUMENT", text)
            return
        else:
            err.emit("INVALID_ARGUMENT", text)
            return
    except Exception as e:
        err.emit("UNKNOWN_ERROR", f"DTW KWS 初始化异常: {e}")
        return

    # 关键词名称
    try:
        if keyword_names_cfg and len(keyword_names_cfg) == len(keywords):
            keyword_names = keyword_names_cfg
        else:
            keyword_names = [os.path.splitext(os.path.basename(p))[0] for p in keywords]
    except Exception as e:
        err.emit("INVALID_ARGUMENT", f"关键词名称处理失败: {e}")
        return

    # VAD
    try:
        vad = VADHelper(sample_rate=sample_rate,
                        aggressiveness=vad_aggr,
                        majority_frames=vad_majority_frames,
                        frame_ms=vad_frame_ms)
    except Exception as e:
        err.emit("VAD_INIT_ERROR", f"WebRTC VAD 初始化失败: {e}")
        return

    emitter = WakeupEmitter(channel=channel)
    q_audio = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            log_warn(f"Audio callback status: {status}")
        try:
            q_audio.put(indata.copy(), block=False)
        except Exception as e:
            log_debug(f"Audio queue full or error: {e}")

    # 音频流
    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_length,
            callback=audio_callback,
            device=device
        )
    except Exception as e:
        err.emit("AUDIO_STREAM_ERROR", f"音频流创建失败: {e}")
        return

    # 主循环
    try:
        log_info(f"采样率: {sample_rate} Hz, 帧长: {frame_length} 样本")
        log_info("启动音频采集，等待唤醒词...（Ctrl+C 退出）")

        with stream:
            pcm_buf_bytes = bytearray()

            while True:
                try:
                    inblock = q_audio.get(timeout=5.0)
                except queue.Empty:
                    err.emit("AUDIO_STREAM_TIMEOUT", "超过 5 秒未收到音频数据", retry_after_ms=2000)
                    continue
                except Exception as e:
                    err.emit("AUDIO_STREAM_ERROR", f"拉取音频数据失败: {e}")
                    break

                try:
                    pcm16 = int16_from_float32(inblock[:, 0])
                except Exception as e:
                    err.emit("PROCESSING_ERROR", f"音频格式转换失败: {e}")
                    continue

                # 更新 VAD 缓冲并判定
                try:
                    pcm_buf_bytes.extend(pcm16.tobytes())
                    vad_flag = vad.is_speech(bytes(pcm_buf_bytes))
                except Exception as e:
                    err.emit("PROCESSING_ERROR", f"VAD 处理失败: {e}")
                    vad_flag = False

                # 切帧送入 KWS（内部自带滑窗）
                try:
                    idx = 0
                    n = len(pcm16)
                    while idx + frame_length <= n:
                        frame = pcm16[idx:idx + frame_length]
                        idx += frame_length
                        res = kws.process(frame)
                        if res >= 0:
                            conf = float(kws.get_last_confidence())
                            try:
                                emitter.emit(
                                    keyword_name=keyword_names[res],
                                    confidence=conf,
                                    vad_flag=vad_flag
                                )
                                log_info(f"唤醒命中：{keyword_names[res]} | conf={conf:.3f}")
                            except Exception as e:
                                err.emit("EMIT_ERROR", f"唤醒事件输出失败: {e}")
                except Exception as e:
                    tb = traceback.format_exc()
                    err.emit("PROCESSING_ERROR", f"KWS 处理失败: {e}\n{tb}")

                # 控制 VAD 缓冲大小（约 1 秒）
                if len(pcm_buf_bytes) > sample_rate * 2:
                    pcm_buf_bytes = pcm_buf_bytes[-sample_rate * 2:]

    except KeyboardInterrupt:
        err.emit("INTERRUPTED", "收到用户中断（Ctrl+C）")
    except Exception as e:
        tb = traceback.format_exc()
        err.emit("UNKNOWN_ERROR", f"主循环异常: {e}\n{tb}")


if __name__ == "__main__":
    main()