# 将原 wakeup.py 的实现移到这里，并去掉主函数部分
# 保留所有的类和工具函数实现
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本程序：纯代码内配置（无命令行参数）
- 传统 KWS（MFCC + DTW）
- WebRTC VAD 多数表决
- Wakeup / Error 事件按 Schema 输出（stdout）
- 文本日志统一走 stderr（通过 log_* 封装）
"""

import json
import queue
import sys
import time
import uuid
from typing import List

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
        "C:/Users/lenovo/Desktop/bababoyi.wav",
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


