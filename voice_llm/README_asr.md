#  FunASR 流式语音识别服务

本项目提供**本地化、低延迟**的中文流式 ASR 服务，基于阿里开源 **FunASR** 的 **Paraformer-zh-streaming** 模型。
同时兼容两种工作模式：

- `asr_service.py`：**切片流式（0.6s）**，稳定低延迟，不做历史回溯纠错；
- `asr_service_streaming.py`：**真流式**（维护 cache，允许模型用后续上下文修正前文），端点刷新后输出最终结果。

> 事件协议与历史版本保持一致：仅从 `stdout` 输出 JSON 行：
> `voice.asr_text.partial` / `voice.asr_text.final` / `core.error_event`。

---

## 1. 环境准备

### 1.1 Python 版本
- Python ≥ 3.8（建议使用 `venv` 或 `conda`）

### 1.2 安装依赖

```bash
pip install -r requirements.txt
```


> 若使用 GPU，请确保 `torch` 与 `torchaudio` 的 CUDA 版本匹配你的驱动。可用 `nvidia-smi` 查看驱动版本。

验证依赖：
```bash
python - <<'PY'
import torch, torchaudio, funasr
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("torchaudio:", torchaudio.__version__)
print("funasr ok")
PY
```

---

## 2. 模型准备（Paraformer-zh-Streaming）

你可以选择 **自动下载** 或 **手动下载到本地目录**：

### 2.1 自动下载
代码中直接：
```python
from funasr import AutoModel
m = AutoModel(model="paraformer-zh-streaming", trust_remote_code=True)
print("cached at:", m.model_path)
```
FunASR 会从 ModelScope/HuggingFace 拉取并缓存在本地用户目录（首次需要联网）。

### 2.2 手动下载
从https://huggingface.co/funasr/paraformer-zh-streaming/tree/main下载
```
am.mvn
config.yaml
configuration.json
model.pt
seg_dict
tokens.json
```
将该目录拷贝到部署机器上，启动时通过 `--model-dir` 指定。


---

## 3. 启动服务

### 3.1 切片流式（稳定低延迟，不做历史修正）
```bash
python asr_service.py --model-dir "/path/to/paraformer-zh-streaming"
# 未提供 --model-dir 时将自动下载并使用缓存模型
```

### 3.2 真流式（可修正前文，端点输出最终结果）
```bash
python asr_service_streaming.py --model-dir "/path/to/paraformer-zh-streaming"
```

### 3.3 日志与输出
- **stderr**：日志；**stdout**：JSON 事件（行分隔）。
- 保存识别事件：
```bash
python asr_service_streaming.py > asr_output.jsonl
```

---

## 4. 事件格式（向后兼容）

### 4.1 部分结果
```json
{"type":"voice.asr_text.partial","ts":1697634534123,"trace_id":"a8c1d2f...","source":"funasr.streaming.paraformer-zh","text":"你好","confidence":0.90}
```

### 4.2 最终结果
```json
{"type":"voice.asr_text.final","ts":1697634535456,"trace_id":"a8c1d2f...","source":"funasr.streaming.paraformer-zh","text":"你好，世界","confidence":0.95}
```

### 4.3 错误事件
```json
{"type":"core.error_event","ts":1697634590230,"trace_id":"b9ac...","source":"voice.asr","error_code":"infer_partial_failed","message":"RuntimeError: ..."}
```

---

## 5. 可调参数（代码顶部 CONFIG）
- `chunk_sec`：麦克风切片时长（默认 0.6s）
- `endpoint_silence_ms`：静音多长判定一句话结束（默认 800ms）
- `chunk_size`：真流式下的 `[chunk, encoder_look_back, decoder_look_back]`（默认 `[5,10,5]`）
- `hotwords`：热词（真流式时传入 `generate()`）

---

## 6. 常见问题（FAQ）

- **`ModuleNotFoundError: No module named 'torch'`**  
  安装 `torch`（CPU 或 CUDA 版）；版本需与 `torchaudio` 匹配。

- **`No module named 'torchaudio'`**  
  安装 `torchaudio`，并与 `torch` 版本匹配。

- **拉取模型失败（网络受限）**  
  使用“手动下载”方式，将完整模型目录（含配置文件）拷贝到部署机，运行时用 `--model-dir` 指定。

- **想减少延迟**  
  适度减小 `chunk_sec`（例如 0.4s），或在 GPU 上运行；注意过小会增加 CPU 使用率。

---

