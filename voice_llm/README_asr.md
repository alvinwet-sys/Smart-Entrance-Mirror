#  SenseVoice 实时语音识别与唤醒系统

本项目提供一个**本地化、低延迟的流式语音识别 (ASR)** 与 **关键词唤醒检测 (Wakeup)** 框架。  
结合 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) 的 **SenseVoice 模型** 与 **WebRTC VAD**，可在无网络环境下实现实时语音到文本识别与自定义唤醒词检测。

---

##  模块概览

| 文件 | 功能 |
|------|------|
| `asr_service.py` | 流式 ASR 服务（SenseVoice + WebRTC VAD）输出部分/最终识别事件 |
| `wakeup.py` | 语音唤醒检测（MFCC + DTW + WebRTC VAD）输出唤醒事件 |

---

## 功能简介

### `asr_service.py` — 实时语音识别服务
- 采集麦克风音频；
- WebRTC VAD 控制语音段；
- Sherpa-ONNX SenseVoice 模型识别；
- 输出 JSON 格式识别事件（partial/final）；
- 错误事件统一格式化输出。

输出示例：
```json
{"type": "voice.asr_text.partial", "ts": 1697634534.12, "trace_id": "a8c1d2f...", "source": "voice", "text": "你好", "confidence": 0.92, "lang": "zh-CN"}
{"type": "voice.asr_text.final", "ts": 1697634535.45, "trace_id": "a8c1d2f...", "source": "voice", "text": "你好，世界", "confidence": 0.94, "lang": "zh-CN"}
```

---

### `wakeup.py` — 离线语音唤醒检测
- 提供一个或多个关键词模板 (`.wav`)；
- 基于 **MFCC + DTW 动态时间规整** 实现关键词匹配；
- 使用 WebRTC VAD 辅助检测有效语音；
- 输出唤醒事件 JSON。

输出示例：
```json
{"ts": 1697634536.42, "trace_id": "7b9f1e...", "source": "voice", "keyword": "你好小镜", "confidence": 0.88, "channel": "near", "vad": true}
```

---

## 环境准备

### 1️⃣ Python 环境
要求：
- Python ≥ 3.8  
- 建议使用 `virtualenv` 或 `conda`

创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate   # Windows 用 venv\Scripts\activate
```

---

### 2️⃣ 安装依赖

```bash
pip install sounddevice numpy webrtcvad librosa sherpa-onnx
```

如果使用 GPU：
```bash
pip install sherpa-onnx[cuda]
```

---

### 3️⃣ 准备 SenseVoice 模型

下载官方 ONNX 模型包：
> https://github.com/k2-fsa/sherpa-onnx/releases

模型目录需包含以下文件：

```
encoder.onnx
decoder.onnx
joiner.onnx
tokens.txt
```

设置环境变量：
```bash
export SENSEVOICE_MODEL_DIR=/path/to/sensevoice_small_onnx
export ASR_PROVIDER=cpu           # 或 cuda
export ASR_THREADS=4
export ASR_FRAME_MS=20
export ASR_VAD_LEVEL=2
export ASR_PARTIAL_MS=150
export ASR_ENDPOINT_MS=600
```

---

### 4️⃣ 准备唤醒词模板

将一到多个 `.wav` 文件放入 `templates/` 目录，例如：
```
templates/kw1.wav
```

修改 `wakeup.py` 顶部的配置：
```python
CONFIG = {
    "keywords": ["./templates/kw1.wav"],
    "keyword_names": ["你好小镜"],
    "dtw_threshold": 250.0,
    ...
}
```

---

## 使用说明

### 启动 ASR 服务
```bash
python asr_service.py
```

日志输出到 `stderr`，识别事件输出到 `stdout`。  
你可以将结果保存：
```bash
python asr_service.py > asr_output.jsonl
```

---

### 启动唤醒检测器
```bash
python wakeup.py
```

运行后，会持续监听麦克风输入，当检测到唤醒词（例如 “你好小镜”）时，输出唤醒事件。

---

##  配置参数说明

### ASR 服务 (`asr_service.py`)

| 环境变量 | 默认值 | 说明 |
|-----------|---------|------|
| `SENSEVOICE_MODEL_DIR` | `/path/to/sensevoice_small_onnx_local` | 模型目录 |
| `ASR_PROVIDER` | `cpu` | 后端设备，可选 `cpu` / `cuda` |
| `ASR_THREADS` | 4 | 模型解码线程数 |
| `ASR_VAD_LEVEL` | 2 | 语音检测灵敏度（0~3） |
| `ASR_PARTIAL_MS` | 150 | 部分结果输出间隔（毫秒） |
| `ASR_ENDPOINT_MS` | 600 | 静音多久认为一句话结束 |

---

### 唤醒检测 (`wakeup.py`)

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `keywords` | `[./templates/kw1.wav]` | 模板路径 |
| `dtw_threshold` | 250.0 | 匹配阈值（越小越严格） |
| `max_window_sec` | 1.0 | 最大匹配窗口 |
| `min_window_sec` | 0.35 | 最小匹配窗口 |
| `cooldown_sec` | 0.8 | 触发后冷却期 |
| `vad_aggressiveness` | 2 | WebRTC VAD 灵敏度 |
| `frame_length` | 512 | 每帧采样点数 |
| `sample_rate` | 16000 | 采样率 |

---

##  示例集成

你可以将两个模块结合使用，例如：
```bash
python wakeup.py | python asr_service.py
```
实现：
> 唤醒检测 → 触发语音识别 → 输出实时识别结果。

---

##  调试与日志

- 所有日志输出到 `stderr`
- 所有事件以 JSON 格式输出到 `stdout`
- 可开启 debug 模式：
  ```python
  CONFIG["log_enable_debug"] = True
  ```
- 如果没有检测到麦克风，请运行：
  ```bash
  python -m sounddevice
  ```

---

##  错误事件格式

所有错误事件遵循统一 Schema：
```json
{
  "ts": 1697634590.23,
  "trace_id": "a9b2f7...",
  "source": "voice.asr",
  "error_code": "AUDIO_STREAM_TIMEOUT",
  "message": "超过5秒未收到音频数据",
  "retry_after_ms": 2000
}
```

---

##  参考项目
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [WebRTC Voice Activity Detection](https://github.com/wiseman/py-webrtcvad)
- [Librosa](https://librosa.org/)

---

