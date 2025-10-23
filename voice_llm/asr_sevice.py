#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化流式 ASR 服务包装器（Stub）。
- 纯代码内配置，无命令行参数
- 简易 VAD 门控并输出部分/最终识别事件
- 标准输出仅写入 JSON 事件，标准错误负责日志
"""

from __future__ import annotations

import json
import queue
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Iterator, List, NamedTuple, Optional

try:
    import numpy as np
    import sounddevice as sd
except ImportError as exc:
    sys.stderr.write(f"缺少必要的音频依赖: {exc}\n")
    sys.stderr.write("请运行: pip install -r requirements.txt\n")
    sys.exit(1)

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore


CONFIG = {
    "sample_rate": 16000,
    "chunk_seconds": 0.6,
    "stride_seconds": 0.2,
    "lang": "zh-CN",
    "device": None,
    "max_queue_seconds": 5.0,
    "logging": {
        "with_time": True,
        "debug": False,
    },
    "vad": {
        "energy_threshold": 0.015,
        "hangover_frames": 3,
    },
    "partial_interval": 0.2,
    "final_silence": 0.6,
}

ASR_REQUIRED_FIELDS = ("ts", "trace_id", "source", "text", "confidence")
ERROR_REQUIRED_FIELDS = ("ts", "trace_id", "source", "error_code", "message")


def _now_str() -> str:
    if not CONFIG["logging"].get("with_time", True):
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(level: str, message: str) -> None:
    prefix = f"[{level}] " if level else ""
    stamp = _now_str()
    line = f"{prefix}{message}" if not stamp else f"{stamp} {prefix}{message}"
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def log_info(message: str) -> None:
    _log("INFO", message)


def log_warn(message: str) -> None:
    _log("WARN", message)


def log_error(message: str) -> None:
    _log("ERROR", message)


def log_debug(message: str) -> None:
    if CONFIG["logging"].get("debug"):
        _log("DEBUG", message)


def now_ts() -> float:
    return time.time()


def gen_trace_id() -> str:
    return uuid.uuid4().hex


def int16_from_float32(chunk: np.ndarray) -> np.ndarray:
    """Convert float32 PCM [-1,1] to int16."""
    chunk = np.clip(chunk, -1.0, 1.0)
    return (chunk * 32767.0).astype(np.int16)


def emit_event(event_name: str, payload: dict) -> None:
    """Emit a single JSON event to stdout after basic validation."""
    if event_name.startswith("voice.asr_text"):
        required = ASR_REQUIRED_FIELDS
    elif event_name == "core.error_event":
        required = ERROR_REQUIRED_FIELDS
    else:
        required = ()

    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields {missing} for {event_name}")

    record = {"type": event_name}
    record.update(payload)
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@dataclass
class Frame:
    pcm16: np.ndarray
    timestamp: float
    duration: float
    is_voiced: bool
    silence: float


class ASRResult(NamedTuple):
    timestamp: float
    trace_id: str
    text: str
    confidence: float
    is_final: bool


class ASREngine:
    """Interface for streaming ASR engines."""

    def transcribe_stream(self, frames: Iterable[Frame]) -> Iterator[ASRResult]:
        raise NotImplementedError


class StubASREngine(ASREngine):
    """Deterministic stub that mimics incremental decoding."""

    def __init__(self, sample_rate: int, partial_interval: float, final_silence: float):
        self.sample_rate = sample_rate
        self.min_partial_interval = partial_interval
        self.final_silence = final_silence
        self.partial_tokens: List[str] = ["正", "在", "识", "别", "语", "音"]
        self.final_templates: List[str] = [
            "模拟语音命令",
            "智能家居控制请求",
            "示例自然语言查询",
        ]
        self._utterance_index = 0

    def transcribe_stream(self, frames: Iterable[Frame]) -> Iterator[ASRResult]:
        trace_id: Optional[str] = None
        voiced_audio: List[np.ndarray] = []
        voiced_duration = 0.0
        last_partial_emit = 0.0
        last_partial_text = ""

        for frame in frames:
            if frame.is_voiced:
                if trace_id is None:
                    trace_id = gen_trace_id()
                    voiced_audio = []
                    voiced_duration = 0.0
                    last_partial_emit = 0.0
                    last_partial_text = ""

                voiced_audio.append(frame.pcm16)
                voiced_duration += frame.duration

                if frame.timestamp - last_partial_emit >= self.min_partial_interval:
                    text = self._make_partial_text(voiced_duration)
                    if text and text != last_partial_text:
                        confidence = self._estimate_confidence(voiced_audio)
                        yield ASRResult(
                            timestamp=frame.timestamp,
                            trace_id=trace_id,
                            text=text,
                            confidence=confidence,
                            is_final=False,
                        )
                        last_partial_emit = frame.timestamp
                        last_partial_text = text

            elif trace_id is not None:
                if frame.silence >= self.final_silence:
                    final_text = self._make_final_text(voiced_audio)
                    confidence = min(0.99, self._estimate_confidence(voiced_audio) + 0.05)
                    confidence = round(confidence, 3)
                    yield ASRResult(
                        timestamp=frame.timestamp,
                        trace_id=trace_id,
                        text=final_text,
                        confidence=confidence,
                        is_final=True,
                    )
                    trace_id = None
                    voiced_audio = []
                    voiced_duration = 0.0
                    last_partial_emit = 0.0
                    last_partial_text = ""
                    self._utterance_index += 1

        # Allow graceful shutdown: finalize lingering audio if loop exits.
        if trace_id and voiced_audio:
            final_text = self._make_final_text(voiced_audio)
            confidence = min(0.99, self._estimate_confidence(voiced_audio) + 0.05)
            yield ASRResult(
                timestamp=now_ts(),
                trace_id=trace_id,
                text=final_text,
                confidence=round(confidence, 3),
                is_final=True,
            )

    def _make_partial_text(self, voiced_duration: float) -> str:
        tokens_count = min(
            len(self.partial_tokens),
            max(1, int(voiced_duration / 0.35)),
        )
        return "".join(self.partial_tokens[:tokens_count])

    def _make_final_text(self, audio_frames: List[np.ndarray]) -> str:
        template = self.final_templates[self._utterance_index % len(self.final_templates)]
        if audio_frames:
            total_ms = int(len(np.concatenate(audio_frames)) / self.sample_rate * 1000)
            return f"{template}（{total_ms}ms）"
        return f"{template}（stub）"

    def _estimate_confidence(self, audio_frames: List[np.ndarray]) -> float:
        if not audio_frames:
            return 0.6
        stacked = np.concatenate(audio_frames).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(stacked ** 2)))
        confidence = 0.55 + min(0.35, rms * 2.2)
        return round(max(0.55, min(confidence, 0.95)), 3)


class SimpleVADGate:
    """Energy-based VAD with optional WebRTC support."""

    def __init__(self, sample_rate: int, threshold: float, hangover_frames: int):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.hangover_frames = max(0, hangover_frames)
        self._hangover = 0
        self._webrtc_vad = None
        if webrtcvad is not None:
            try:
                self._webrtc_vad = webrtcvad.Vad(2)
            except Exception as exc:  # pragma: no cover - defensive
                log_warn(f"WebRTC VAD 初始化失败，fallback：{exc}")
                self._webrtc_vad = None

    def is_voiced(self, pcm16: np.ndarray) -> bool:
        if pcm16.size == 0:
            return False

        if self._webrtc_vad is not None:
            frame_size = int(self.sample_rate * 0.02)
            bytes_per_sample = 2
            pcm_bytes = pcm16.tobytes()
            voiced = False
            for offset in range(0, len(pcm16) - frame_size + 1, frame_size):
                chunk = pcm_bytes[offset * bytes_per_sample : (offset + frame_size) * bytes_per_sample]
                if self._webrtc_vad.is_speech(chunk, self.sample_rate):
                    voiced = True
                    break
        else:
            normalized = pcm16.astype(np.float32) / 32768.0
            energy = float(np.mean(np.abs(normalized)))
            voiced = energy >= self.threshold

        if voiced:
            self._hangover = self.hangover_frames
            return True

        if self._hangover > 0:
            self._hangover -= 1
            return True

        return False


class AudioSource:
    """Microphone reader producing stride-sized PCM frames."""

    def __init__(
        self,
        sample_rate: int,
        stride_seconds: float,
        device: Optional[int],
        max_queue_seconds: float,
    ):
        self.sample_rate = sample_rate
        self.stride_seconds = stride_seconds
        self.blocksize = max(1, int(sample_rate * stride_seconds))
        self.device = device
        max_items = max(1, int(max_queue_seconds / stride_seconds) + 1)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_items)

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        if status:
            log_warn(f"音频回调状态: {status}")
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            log_warn("音频队列已满，丢弃一帧")

    def frames(self) -> Iterator[np.ndarray]:
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._audio_callback,
            device=self.device,
        )

        try:
            stream.start()
            log_info(
                f"ASR 音频输入启动 (sr={self.sample_rate}, stride={self.blocksize} samples)"
            )
            while True:
                try:
                    chunk = self._queue.get(timeout=max(1.0, CONFIG["stride_seconds"] * 5))
                except queue.Empty as exc:
                    raise RuntimeError("音频输入超时") from exc
                if chunk.size == 0:
                    continue
                yield int16_from_float32(chunk[:, 0])
        finally:
            stream.stop()
            stream.close()
            log_info("ASR 音频输入已关闭")


class ASRService:
    """Glue logic from audio frames → VAD → stub ASR → event emission."""

    def __init__(
        self,
        audio_source: AudioSource,
        vad_gate: SimpleVADGate,
        engine: ASREngine,
        config: dict,
    ):
        self.audio_source = audio_source
        self.vad = vad_gate
        self.engine = engine
        self.config = config
        self._silence = 0.0
        self._error_emitted = False

    def run(self) -> None:
        log_info("ASR 服务启动 (Stub 引擎)")
        frames_iter = self._frame_generator()
        results = self.engine.transcribe_stream(frames_iter)

        try:
            for result in results:
                event_name = (
                    "voice.asr_text.final" if result.is_final else "voice.asr_text.partial"
                )
                payload = {
                    "ts": result.timestamp,
                    "trace_id": result.trace_id,
                    "source": "voice",
                    "text": result.text,
                    "confidence": float(min(1.0, max(0.0, result.confidence))),
                    "lang": self.config["lang"],
                }
                emit_event(event_name, payload)
        except KeyboardInterrupt:
            log_info("检测到退出请求，停止 ASR 服务")
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_error("asr.stream_failure", str(exc), retry_after_ms=2000)

    def _frame_generator(self) -> Iterator[Frame]:
        stride_duration = self.config["stride_seconds"]
        for pcm16 in self.audio_source.frames():
            timestamp = now_ts()
            voiced = self.vad.is_voiced(pcm16)
            if voiced:
                self._silence = 0.0
            else:
                self._silence += stride_duration
            yield Frame(
                pcm16=pcm16,
                timestamp=timestamp,
                duration=stride_duration,
                is_voiced=voiced,
                silence=self._silence,
            )

    def _handle_error(self, error_code: str, message: str, retry_after_ms: Optional[int] = None) -> None:
        if self._error_emitted:
            log_error(f"{error_code}: {message}")
            sys.exit(1)

        payload = {
            "ts": now_ts(),
            "trace_id": gen_trace_id(),
            "source": "voice.asr",
            "error_code": error_code,
            "message": message,
        }
        if retry_after_ms is not None:
            payload["retry_after_ms"] = max(0, int(retry_after_ms))

        try:
            emit_event("core.error_event", payload)
        except Exception as exc:  # pragma: no cover - defensive
            log_error(f"错误事件发送失败: {exc}")

        log_error(f"{error_code}: {message}")
        self._error_emitted = True
        sys.exit(1)


def main() -> None:
    cfg = CONFIG
    audio_source = AudioSource(
        sample_rate=cfg["sample_rate"],
        stride_seconds=cfg["stride_seconds"],
        device=cfg["device"],
        max_queue_seconds=cfg["max_queue_seconds"],
    )
    vad_gate = SimpleVADGate(
        sample_rate=cfg["sample_rate"],
        threshold=cfg["vad"]["energy_threshold"],
        hangover_frames=cfg["vad"]["hangover_frames"],
    )
    engine = StubASREngine(
        sample_rate=cfg["sample_rate"],
        partial_interval=cfg["partial_interval"],
        final_silence=cfg["final_silence"],
    )
    service = ASRService(audio_source, vad_gate, engine, cfg)

    try:
        service.run()
    except KeyboardInterrupt:
        log_info("ASR 服务已结束")


if __name__ == "__main__":
    main()
