#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化流式 ASR 服务包装器（SenseVoice）。
- sherpa-onnx 在线识别
- WebRTC VAD 门控并输出部分/最终识别事件
- 标准输出仅写入 JSON 事件，标准错误负责日志
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from typing import Callable, Iterator, Optional

try:
    import numpy as np
    import sounddevice as sd
except ImportError as exc:
    sys.stderr.write(f"缺少必要的音频依赖: {exc}\n")
    sys.stderr.write("请运行: pip install -r requirements.txt\n")
    sys.exit(1)

try:
    import webrtcvad  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore

try:
    import sherpa_onnx  # type: ignore
except ImportError as exc:
    sys.stderr.write(f"缺少语音识别依赖: {exc}\n")
    sys.stderr.write("请运行: pip install -r voice_llm/requirements\n")
    sys.exit(1)

CONFIG = {
    "sample_rate": 16000,
    "stride_seconds": 0.2,
    "lang": "zh-CN",
    "device": None,
    "max_queue_seconds": 5.0,
    "logging": {
        "with_time": True,
        "debug": False,
    },
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
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_items)

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
                    chunk = self._queue.get(timeout=max(1.0, self.stride_seconds * 5))
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
    """Glue logic from audio frames → SenseVoice → event emission."""

    def __init__(
        self,
        engine: "SenseVoiceEngine",
        config: Optional[dict] = None,
        audio_source: Optional[AudioSource] = None,
    ):
        self.engine = engine
        self.config = dict(CONFIG)
        if config is not None:
            self.config.update(config)
        self.audio_source = audio_source or AudioSource(
            sample_rate=self.config["sample_rate"],
            stride_seconds=self.config["stride_seconds"],
            device=self.config["device"],
            max_queue_seconds=self.config["max_queue_seconds"],
        )
        self.lang = self.config.get("lang", "zh-CN")
        self._trace_id: Optional[str] = None
        self._last_partial_text: str = ""
        self._error_emitted = False

    def publish_partial(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        ts = now_ts()
        if self._trace_id is None:
            self._trace_id = gen_trace_id()
        if text == self._last_partial_text:
            return
        self._last_partial_text = text
        confidence = float(min(1.0, max(0.0, self.engine.latest_confidence)))

        payload = {
            "ts": ts,
            "trace_id": self._trace_id,
            "source": "voice",
            "text": text,
            "confidence": confidence,
            "lang": self.lang,
        }
        emit_event("voice.asr_text.partial", payload)

    def publish_final(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        ts = now_ts()
        if self._trace_id is None:
            self._trace_id = gen_trace_id()
        confidence = float(min(1.0, max(0.0, self.engine.latest_confidence)))

        payload = {
            "ts": ts,
            "trace_id": self._trace_id,
            "source": "voice",
            "text": text,
            "confidence": confidence,
            "lang": self.lang,
        }
        emit_event("voice.asr_text.final", payload)
        self._trace_id = None
        self._last_partial_text = ""

    def run(self) -> None:
        log_info("ASR 服务启动 (SenseVoice 引擎)")
        self.engine.start()

        try:
            for pcm16 in self.audio_source.frames():
                self.engine.push_audio(pcm16.tobytes())
        except KeyboardInterrupt:
            log_info("检测到退出请求，停止 ASR 服务")
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_error("asr.stream_failure", str(exc), retry_after_ms=2000)
        finally:
            self.engine.stop()
            log_info("ASR 服务已结束")

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


class SenseVoiceEngine:
    """Streaming ASR engine backed by sherpa-onnx SenseVoice models."""

    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        vad_aggressiveness: int = 2,
        partial_interval_ms: int = 150,
        endpoint_silence_ms: int = 600,
        provider: str = "cpu",
        num_threads: int = 4,
    ):
        if webrtcvad is None:
            raise RuntimeError("未安装 webrtcvad，请运行 pip install -r voice_llm/requirements")

        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.frame_ms = int(frame_ms)
        if self.frame_ms not in (10, 20, 30):
            allowed = min((10, 20, 30), key=lambda v: abs(v - self.frame_ms))
            log_warn(f"WebRTC VAD 仅支持 10/20/30ms 帧，自动调整为 {allowed}ms")
            self.frame_ms = allowed
        self.frame_seconds = self.frame_ms / 1000.0
        self.frame_samples = max(1, int(self.sample_rate * self.frame_seconds))
        self._frame_bytes = self.frame_samples * 2

        self.partial_interval = max(0.05, partial_interval_ms / 1000.0)
        self.endpoint_silence = max(self.frame_seconds, endpoint_silence_ms / 1000.0)
        self._vad = webrtcvad.Vad(max(0, min(3, vad_aggressiveness)))

        frames_per_second = max(1, int(round(1.0 / self.frame_seconds)))
        queue_size = max(50, frames_per_second * 5)
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=queue_size)
        self._pending = bytearray()

        self._stop_event = threading.Event()
        self._decode_thread: Optional[threading.Thread] = None
        self._partial_callback: Callable[[str], None] = lambda _: None
        self._final_callback: Callable[[str], None] = lambda _: None

        self._last_partial_emit = 0.0
        self._last_partial_text = ""
        self._utterance_active = False
        self._current_silence = 0.0
        self._last_voiced_time = time.monotonic()
        self._energy_sum = 0.0
        self._energy_count = 0
        self._latest_confidence = 0.6
        self._last_drop_log = 0.0

        self.provider = provider
        self.num_threads = num_threads
        self._recognizer, self._stream = self._create_recognizer(model_dir, provider, num_threads)
        self._stream_lock = threading.Lock()

    @property
    def latest_confidence(self) -> float:
        return self._latest_confidence

    def on_partial(self, callback: Callable[[str], None]) -> None:
        self._partial_callback = callback

    def on_final(self, callback: Callable[[str], None]) -> None:
        self._final_callback = callback

    def start(self) -> None:
        if self._decode_thread and self._decode_thread.is_alive():
            return
        self._stop_event.clear()
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            name="SenseVoiceDecoder",
            daemon=True,
        )
        self._decode_thread.start()
        log_info("SenseVoice 解码线程已启动")

    def stop(self) -> None:
        self._stop_event.set()
        if self._decode_thread and self._decode_thread.is_alive():
            inserted = False
            while not inserted:
                try:
                    self._queue.put_nowait(None)
                    inserted = True
                except queue.Full:
                    try:
                        _ = self._queue.get_nowait()
                    except queue.Empty:
                        break
            self._decode_thread.join(timeout=2.0)
        self._decode_thread = None
        if self._utterance_active:
            self._finalize_utterance(time.monotonic())
        log_info("SenseVoice 解码线程已停止")

    def push_audio(self, pcm16le_bytes: bytes) -> None:
        if not pcm16le_bytes or self._stop_event.is_set():
            return

        self._pending.extend(pcm16le_bytes)
        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    now = time.monotonic()
                    if now - self._last_drop_log > 1.0:
                        log_warn("SenseVoice 队列已满，丢弃一帧音频")
                        self._last_drop_log = now

    def _decode_loop(self) -> None:
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                self._check_endpoint(force=True)
                break

            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._check_endpoint()
                continue

            if frame is None:
                self._check_endpoint(force=True)
                break

            try:
                self._handle_frame(frame)
            except Exception as exc:  # pragma: no cover - defensive
                log_error(f"SenseVoice 解码异常: {exc}")

    def _handle_frame(self, frame: bytes) -> None:
        now = time.monotonic()
        try:
            voiced = self._vad.is_speech(frame, self.sample_rate)
        except Exception as exc:
            log_warn(f"WebRTC VAD 处理失败，假定为语音: {exc}")
            voiced = True

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        self._accept_waveform(samples)
        text = self._decode_text(finalize=False)

        if voiced:
            if not self._utterance_active:
                self._utterance_active = True
                self._energy_sum = 0.0
                self._energy_count = 0
                self._current_silence = 0.0
            self._last_voiced_time = now
            self._current_silence = 0.0
            self._update_energy(samples)
            self._emit_partial(now, text)
        else:
            if self._utterance_active:
                self._current_silence += self.frame_seconds
                if self._current_silence >= self.endpoint_silence:
                    self._finalize_utterance(now)

    def _accept_waveform(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self._stream_lock:
            self._stream.accept_waveform(self.sample_rate, samples)

    def _decode_text(self, finalize: bool) -> str:
        with self._stream_lock:
            if finalize and hasattr(self._stream, "input_finished"):
                self._stream.input_finished()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            result = self._recognizer.get_result(self._stream)
        return (result.text or "").strip()

    def _emit_partial(self, now: float, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text == self._last_partial_text and now - self._last_partial_emit < self.partial_interval:
            return
        if now - self._last_partial_emit < self.partial_interval:
            return

        self._latest_confidence = self._estimate_confidence()
        self._last_partial_emit = now
        self._last_partial_text = text
        self._invoke_callback(self._partial_callback, text)

    def _finalize_utterance(self, now: float) -> None:
        if not self._utterance_active:
            return

        text = self._decode_text(finalize=True)
        text = text.strip()
        if text:
            self._latest_confidence = self._estimate_confidence()
            self._invoke_callback(self._final_callback, text)
        self._reset_stream(now)

    def _reset_stream(self, now: float) -> None:
        with self._stream_lock:
            self._stream = self._recognizer.create_stream()

        self._utterance_active = False
        self._current_silence = 0.0
        self._last_partial_emit = 0.0
        self._last_partial_text = ""
        self._energy_sum = 0.0
        self._energy_count = 0
        self._latest_confidence = 0.6
        self._last_voiced_time = now

    def _check_endpoint(self, force: bool = False) -> None:
        if not self._utterance_active:
            return
        now = time.monotonic()
        elapsed = now - self._last_voiced_time
        if force or elapsed >= self.endpoint_silence:
            self._finalize_utterance(now)

    def _update_energy(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        energy = float(np.sum(samples * samples))
        self._energy_sum += energy
        self._energy_count += samples.size

    def _estimate_confidence(self) -> float:
        if self._energy_count == 0:
            return 0.6
        rms = math.sqrt(self._energy_sum / self._energy_count)
        confidence = 0.55 + min(0.35, rms * 2.2)
        return round(max(0.55, min(confidence, 0.98)), 3)

    def _invoke_callback(self, callback: Callable[[str], None], text: str) -> None:
        try:
            callback(text)
        except Exception as exc:  # pragma: no cover - defensive
            log_error(f"ASR 回调执行失败: {exc}")

    def _create_recognizer(self, model_dir: str, provider: str, num_threads: int):
        encoder = os.path.join(model_dir, "encoder.onnx")
        decoder = os.path.join(model_dir, "decoder.onnx")
        joiner = os.path.join(model_dir, "joiner.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        missing = [path for path in (encoder, decoder, joiner, tokens) if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"SenseVoice 模型文件缺失: {', '.join(missing)}")

        model_config = sherpa_onnx.OnlineModelConfig()
        model_config.transducer.encoder = encoder
        model_config.transducer.decoder = decoder
        model_config.transducer.joiner = joiner
        model_config.tokens = tokens
        if hasattr(model_config, "sample_rate"):
            model_config.sample_rate = self.sample_rate
        if hasattr(model_config, "feature_dim"):
            model_config.feature_dim = 80
        if hasattr(model_config, "num_threads"):
            model_config.num_threads = num_threads
        if hasattr(model_config, "provider"):
            model_config.provider = provider
        if hasattr(model_config, "use_gpu"):
            model_config.use_gpu = provider == "cuda"

        recognizer_config = sherpa_onnx.OnlineRecognizerConfig()
        recognizer_config.model = model_config
        recognizer_config.decoding_method = "greedy_search"
        if hasattr(recognizer_config, "enable_endpoint"):
            recognizer_config.enable_endpoint = False

        recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
        stream = recognizer.create_stream()
        return recognizer, stream


def main() -> None:
    import os

    model_dir = os.getenv("SENSEVOICE_MODEL_DIR", "/path/to/sensevoice_small_onnx_local")
    provider = os.getenv("ASR_PROVIDER", "cpu")  # "cpu" or "cuda"
    num_threads = int(os.getenv("ASR_THREADS", "4"))
    frame_ms = int(os.getenv("ASR_FRAME_MS", "20"))
    vad_level = int(os.getenv("ASR_VAD_LEVEL", "2"))
    partial_ms = int(os.getenv("ASR_PARTIAL_MS", "150"))
    endpoint_ms = int(os.getenv("ASR_ENDPOINT_MS", "600"))

    engine = SenseVoiceEngine(
        model_dir=model_dir,
        sample_rate=16000,
        frame_ms=frame_ms,
        vad_aggressiveness=vad_level,
        partial_interval_ms=partial_ms,
        endpoint_silence_ms=endpoint_ms,
        provider=provider,
        num_threads=num_threads,
    )

    service = ASRService(engine=engine)
    engine.on_partial(lambda txt: service.publish_partial(txt))
    engine.on_final(lambda txt: service.publish_final(txt))
    engine.start()
    service.run()


if __name__ == "__main__":
    main()
