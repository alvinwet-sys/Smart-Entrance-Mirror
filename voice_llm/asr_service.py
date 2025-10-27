
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_service.py — Streaming ASR using FunASR Paraformer (zh-streaming)
- 依赖：funasr, modelscope, pyaudio, numpy, soundfile（写临时wav）, webrtcvad(可选)
- 输入：麦克风（PyAudio），每 0.6s 切片；按段输出部分/最终结果
- JSON 接口：
  * voice.asr_text.partial
  * voice.asr_text.final
  * core.error_event

用法：
  pip install -U funasr modelscope pyaudio numpy soundfile webrtcvad
  python asr_service.py --model-dir /path/to/paraformer-zh-streaming  # 可选，不传则自动下载
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import tempfile
import time
import uuid
from typing import Optional, Iterator

import numpy as np
import soundfile as sf

try:
    import pyaudio  # type: ignore
except Exception as e:
    sys.stderr.write(f"[asr] 缺少 pyaudio：{e}\n请先安装：pip install pyaudio\n")
    sys.exit(1)

try:
    import webrtcvad  # type: ignore
    _vad_available = True
except Exception:
    webrtcvad = None
    _vad_available = False

try:
    import torch  # 仅用于判断是否可用 CUDA
    _cuda = torch.cuda.is_available()
except Exception:
    _cuda = False

try:
    from funasr import AutoModel  # type: ignore
except Exception as e:
    sys.stderr.write(f"[asr] 缺少 FunASR 依赖：{e}\n请先安装：pip install -U funasr modelscope\n")
    sys.exit(1)


# ===================== 可调参数 =====================
CONFIG = {
    "sample_rate": 16000,
    "chunk_sec": 0.6,            # 每次 0.6s 切片
    "max_buffer_sec": 6.0,       # 识别时使用的滚动缓冲最大长度
    "endpoint_silence_ms": 800,  # 判定端点的静音阈值
    "device": "cuda" if _cuda else "cpu",
    "logging": {"with_time": True, "debug": False},
}


ASR_REQUIRED_FIELDS = ("ts", "trace_id", "source", "text", "confidence")
ERROR_REQUIRED_FIELDS = ("ts", "trace_id", "source", "error_code", "message")


# ===================== 工具函数 =====================
def now_ts() -> int:
    return int(time.time() * 1000)


def gen_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def tlog(msg: str) -> None:
    if CONFIG["logging"].get("with_time", True):
        sys.stderr.write(time.strftime("%H:%M:%S ") + msg + "\n")
    else:
        sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def emit_event(event: str, payload: dict) -> None:
    if event.startswith("voice.asr_text"):
        required = ASR_REQUIRED_FIELDS
    elif event == "core.error_event":
        required = ERROR_REQUIRED_FIELDS
    else:
        required = ()
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields {missing} for {event}")
    record = {"type": event}
    record.update(payload)
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ===================== ASR 封装 =====================
class FunASRWrapper:
    """
    轻量封装：维护滚动缓冲；每次将缓冲写临时 WAV 并调用 AutoModel.generate。
    （说明：FunASR 也有真流式接口；这里按你的“0.6s 切片+即时输出”策略实现，
     不做历史回溯纠错，以换取稳定低延迟。）
    """
    def __init__(self, model_dir: Optional[str], device: str):
        if model_dir:
            self.model = AutoModel(model=model_dir, trust_remote_code=True, device=device)
        else:
            # 官方在线模型名（会自动下载到 ~/.cache/modelscope）
            self.model = AutoModel(model="paraformer-zh-streaming", trust_remote_code=True, device=device)
        self.sr = CONFIG["sample_rate"]
        self.buffer = np.zeros(0, dtype=np.float32)

    def append_audio(self, chunk: np.ndarray) -> None:
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        self.buffer = np.concatenate([self.buffer, chunk])
        # 控制滚动缓冲长度，避免推理越来越慢
        max_len = int(CONFIG["max_buffer_sec"] * self.sr)
        if self.buffer.shape[0] > max_len:
            self.buffer = self.buffer[-max_len:]

    def _infer_buffer(self) -> str:
        """将当前缓冲写入临时 WAV，并调用 FunASR 识别"""
        if self.buffer.size == 0:
            return ""
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # 我们用 soundfile 写
            sf.write(wav_path, self.buffer, self.sr, subtype="PCM_16")
            # 兼容不同返回结构：有些版本返回 dict/text，有些返回 list[dict]
            result = self.model.generate(input=[wav_path])
            text = ""
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
            elif isinstance(result, list):
                if len(result) and isinstance(result[0], dict):
                    text = result[0].get("text", "")
                else:
                    text = " ".join(str(x) for x in result)
            else:
                text = str(result)
            return text.strip()
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def recognize_partial(self) -> str:
        return self._infer_buffer()

    def recognize_final(self) -> str:
        text = self._infer_buffer()
        # 清掉缓冲，开始新的段落
        self.buffer = np.zeros(0, dtype=np.float32)
        return text


# ===================== 简单 VAD（webrtcvad） =====================
class SimpleVAD:
    """
    将输入的 int16 PCM 分成 30ms 小帧，统计有声比例；超过阈值判定为语音。
    """
    def __init__(self, sample_rate: int, aggressiveness: int = 2, speech_ratio: float = 0.5):
        self.sr = sample_rate
        self.speech_ratio = speech_ratio
        self.vad = None
        if _vad_available:
            self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, pcm16: bytes) -> bool:
        if not self.vad:
            # 没有 webrtcvad 时，默认都当成语音，避免误砍
            return True
        frame_bytes = int(0.03 * self.sr) * 2  # 30ms, 16-bit
        total = 0
        voiced = 0
        for i in range(0, len(pcm16), frame_bytes):
            frame = pcm16[i: i + frame_bytes]
            if len(frame) < frame_bytes:
                break
            total += 1
            try:
                if self.vad.is_speech(frame, self.sr):
                    voiced += 1
            except Exception:
                pass
        if total == 0:
            return False
        return (voiced / total) >= self.speech_ratio


# ===================== 采集与主循环 =====================
def run_mic(model_dir: Optional[str] = None):
    sr = CONFIG["sample_rate"]
    chunk = int(CONFIG["chunk_sec"] * sr)
    pa = pyaudio.PyAudio()

    # 使用 Int16 采集，后续转 float32 [-1,1]
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=chunk)

    asr = FunASRWrapper(model_dir, CONFIG["device"])
    vad = SimpleVAD(sr)

    tlog(f"[asr] Start mic streaming: sr={sr}, chunk={CONFIG['chunk_sec']}s, device={CONFIG['device']}")
    trace_id = gen_trace_id()
    trailing_sil_ms = 0

    try:
        while True:
            pcm_bytes = stream.read(chunk, exception_on_overflow=False)
            # 复制到 float32
            data16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            mono = (data16.astype(np.float32) / 32768.0).reshape(-1)
            asr.append_audio(mono)

            # partial
            try:
                partial_text = asr.recognize_partial()
                if partial_text:
                    emit_event("voice.asr_text.partial", {
                        "ts": now_ts(),
                        "trace_id": trace_id,
                        "source": "funasr.paraformer-zh-streaming",
                        "text": partial_text,
                        "confidence": 0.9,
                    })
            except Exception as e:
                emit_event("core.error_event", {
                    "ts": now_ts(),
                    "trace_id": trace_id,
                    "source": "voice.asr",
                    "error_code": "infer_partial_failed",
                    "message": f"{type(e).__name__}: {e}",
                })

            # endpointing
            if vad.is_speech(pcm_bytes):
                trailing_sil_ms = 0
            else:
                trailing_sil_ms += int(1000 * len(data16) / sr)

            if trailing_sil_ms >= CONFIG["endpoint_silence_ms"]:
                try:
                    final_text = asr.recognize_final()
                    if final_text:
                        emit_event("voice.asr_text.final", {
                            "ts": now_ts(),
                            "trace_id": trace_id,
                            "source": "funasr.paraformer-zh-streaming",
                            "text": final_text,
                            "confidence": 0.9,
                        })
                    trace_id = gen_trace_id()
                except Exception as e:
                    emit_event("core.error_event", {
                        "ts": now_ts(),
                        "trace_id": trace_id,
                        "source": "voice.asr",
                        "error_code": "infer_final_failed",
                        "message": f"{type(e).__name__}: {e}",
                    })
                finally:
                    trailing_sil_ms = 0
    except KeyboardInterrupt:
        tlog("[asr] Stopped by user")
    except Exception as e:
        emit_event("core.error_event", {
            "ts": now_ts(),
            "trace_id": trace_id,
            "source": "voice.asr",
            "error_code": "runtime_error",
            "message": f"{type(e).__name__}: {e}",
        })
    finally:
        try:
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()


# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser(description="FunASR streaming service (Paraformer zh)")
    ap.add_argument("--model-dir", default=None, help="本地 paraformer-zh-streaming 模型目录；不传则自动下载")
    return ap.parse_args()


def main():
    args = parse_args()
    run_mic(args.model_dir)


if __name__ == "__main__":
    main()
