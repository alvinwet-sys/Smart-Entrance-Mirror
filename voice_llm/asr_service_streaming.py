# python ./voice_llm/asr_service_streaming.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (Recreated) asr_service_streaming.py — True streaming FunASR service
# See docstring in previous attempt for full details.
from __future__ import annotations
import argparse, json, os, sys, time, uuid, tempfile
from typing import Optional
import numpy as np

try:
    import pyaudio  # type: ignore
except Exception as e:
    print(f"[asr] 缺少 pyaudio：{e}", file=sys.stderr)
    raise

try:
    import webrtcvad  # type: ignore
    _vad_available = True
except Exception:
    webrtcvad = None
    _vad_available = False

try:
    import torch
    _cuda = torch.cuda.is_available()
except Exception:
    _cuda = False

try:
    from funasr import AutoModel  # type: ignore
except Exception as e:
    print(f"[asr] 缺少 FunASR 依赖：{e}", file=sys.stderr)
    raise


CONFIG = {
    "sample_rate": 16000,
    "chunk_sec": 0.6,
    "endpoint_silence_ms": 800,
    "device": "cuda" if _cuda else "cpu",
    "chunk_size": [5, 10, 5],
    "hotwords": "",
    "logging": True,
}

ASR_REQUIRED_FIELDS = ("ts", "trace_id", "source", "text", "confidence")
ERROR_REQUIRED_FIELDS = ("ts", "trace_id", "source", "error_code", "message")


def now_ts(): return int(time.time() * 1000)
def gen_trace_id(): return uuid.uuid4().hex[:16]
def tlog(msg: str):
    if CONFIG["logging"]:
        print(time.strftime("%H:%M:%S ") + msg, file=sys.stderr)


def emit_event(event: str, payload: dict):
    if event.startswith("voice.asr_text"):
        required = ASR_REQUIRED_FIELDS
    elif event == "core.error_event":
        required = ERROR_REQUIRED_FIELDS
    else:
        required = ()
    missing = [k for k in required if k not in payload]
    if missing: raise ValueError(f"Missing {missing} for {event}")
    rec = {"type": event}; rec.update(payload)
    print(json.dumps(rec, ensure_ascii=False))


class SimpleVAD:
    def __init__(self, sample_rate: int, aggressiveness: int = 2):
        self.sr = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness) if _vad_available else None
    def is_speech(self, pcm16: bytes) -> bool:
        if not self.vad: return True
        frame_bytes = int(0.03 * self.sr) * 2
        voiced, total = 0, 0
        for i in range(0, len(pcm16), frame_bytes):
            frame = pcm16[i:i+frame_bytes]
            if len(frame) < frame_bytes: break
            total += 1
            try:
                if self.vad.is_speech(frame, self.sr): voiced += 1
            except Exception: pass
        return total > 0 and (voiced / total) >= 0.5


class FunASRStreaming:
    def __init__(self, model_dir: Optional[str], device: str, sr: int):
        if model_dir:
            self.model = AutoModel(model=model_dir, trust_remote_code=True, device=device)
        else:
            self.model = AutoModel(model="paraformer-zh-streaming", trust_remote_code=True, device=device)
        self.sr = sr
        self.cache = None
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_text = ""
    def reset(self):
        self.cache = None
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_text = ""
    def _call_streaming(self, chunk: np.ndarray, is_final: bool) -> str:
        kwargs_list = [
            dict(input=chunk, cache=self.cache, is_final=is_final,
                 chunk_size=CONFIG["chunk_size"], sampling_rate=self.sr, hotwords=CONFIG["hotwords"]),
            dict(input=chunk, cache=self.cache, is_final=is_final,
                 chunk_size=CONFIG["chunk_size"], sample_rate=self.sr, hotwords=CONFIG["hotwords"]),
        ]
        last_exc = None
        for kwargs in kwargs_list:
            try:
                out = self.model.generate(**kwargs)
                text, cache = "", None
                if isinstance(out, list) and out and isinstance(out[0], dict):
                    text = out[0].get("text",""); cache = out[0].get("cache")
                elif isinstance(out, dict):
                    text = out.get("text",""); cache = out.get("cache")
                else:
                    text = str(out)
                self.cache = cache if cache is not None else self.cache
                return text.strip()
            except Exception as e:
                last_exc = e; continue
        raise last_exc if last_exc else RuntimeError("Unknown streaming error")
    def feed(self, chunk_f32: np.ndarray, is_final: bool) -> str:
        try:
            return self._call_streaming(chunk_f32, is_final=is_final)
        except Exception:
            # fallback: temp wav
            import soundfile as sf
            self.buffer = np.concatenate([self.buffer, chunk_f32])
            path = None
            try:
                fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                sf.write(path, self.buffer, self.sr, subtype="PCM_16")
                out = self.model.generate(input=[path])
                if isinstance(out, list) and out and isinstance(out[0], dict):
                    return out[0].get("text","").strip()
                if isinstance(out, dict): return out.get("text","").strip()
                return str(out).strip()
            finally:
                if path and os.path.exists(path):
                    try: os.remove(path)
                    except Exception: pass


def run(model_dir: Optional[str] = None):
    sr = CONFIG["sample_rate"]; frames = int(CONFIG["chunk_sec"] * sr)
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=frames)
    vad = SimpleVAD(sr)
    asr = FunASRStreaming(model_dir, CONFIG["device"], sr)
    trace_id = gen_trace_id(); trailing_ms = 0
    tlog(f"[asr] Start (true streaming) sr={sr}, chunk={CONFIG['chunk_sec']}s, device={CONFIG['device']}")
    try:
        while True:
            pcm = stream.read(frames, exception_on_overflow=False)
            data16 = np.frombuffer(pcm, dtype=np.int16)
            chunk_f32 = (data16.astype(np.float32) / 32768.0).reshape(-1)
            try:
                text = asr.feed(chunk_f32, is_final=False)
                if text and text != asr.last_text:
                    asr.last_text = text
                    emit_event("voice.asr_text.partial", {
                        "ts": now_ts(), "trace_id": trace_id, "source": "funasr.streaming.paraformer-zh",
                        "text": text, "confidence": 0.9,
                    })
            except Exception as e:
                emit_event("core.error_event", {
                    "ts": now_ts(), "trace_id": trace_id, "source": "voice.asr",
                    "error_code": "infer_partial_failed", "message": f"{type(e).__name__}: {e}",
                })
            if vad.is_speech(pcm):
                trailing_ms = 0
            else:
                trailing_ms += int(len(data16) / sr * 1000)
            if trailing_ms >= CONFIG["endpoint_silence_ms"]:
                try:
                    final_text = asr.feed(np.zeros(0, dtype=np.float32), is_final=True) or asr.last_text
                    if final_text:
                        emit_event("voice.asr_text.final", {
                            "ts": now_ts(), "trace_id": trace_id, "source": "funasr.streaming.paraformer-zh",
                            "text": final_text, "confidence": 0.95,
                        })
                except Exception as e:
                    emit_event("core.error_event", {
                        "ts": now_ts(), "trace_id": trace_id, "source": "voice.asr",
                        "error_code": "infer_final_failed", "message": f"{type(e).__name__}: {e}",
                    })
                finally:
                    asr.reset(); trace_id = gen_trace_id(); trailing_ms = 0
    except KeyboardInterrupt:
        tlog("[asr] Stopped")
    finally:
        try: stream.stop_stream(); stream.close()
        finally: pa.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="本地 paraformer-zh-streaming 模型目录（不传则自动下载）")
    args = ap.parse_args()
    run(args.model_dir)


if __name__ == "__main__":
    main()
