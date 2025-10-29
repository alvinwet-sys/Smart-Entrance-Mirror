# TTS语音合成模块

## 1. 模块概览

### 功能描述
本模块是基于百度语音API的TTS（文本转语音）服务，负责将文本内容转换为语音并播放。主要功能包括：
- **输入**：文本内容、播放优先级、语音风格等参数
- **输出**：语音播放，并返回播放状态事件
- **核心能力**：支持多任务队列管理、优先级播放、播放中断等特性

### 技术选型
- **核心依赖**：百度语音API（AipSpeech）
- **多线程处理**：Python threading模块实现播放队列管理
- **跨平台播放**：使用系统原生播放器（Windows Media Player/afplay/aplay）
- **配置管理**：JSON配置文件存储API凭证

## 2. 环境配置

### 依赖项列表
```bash
pip install baidu-aip
pip install jsonschema
pip install playsound  # 可选，作为备用播放方案
```

### API配置步骤
1. 首次运行时会提示输入百度语音服务的API凭证：
   - APP_ID：百度语音应用ID
   - API_KEY：百度语音API密钥
   - SECRET_KEY：百度语音安全密钥
2. 配置信息将自动保存到 `~/.baidu_tts_config.json`
3. 在Unix-like系统上会自动设置配置文件权限为600

## 3. 接口规范

### 对外接口

#### TTSModule类初始化
```python
tts = TTSModule(event_callback=None, config_file=None)
```
- `event_callback`: 事件回调函数，接收JSON格式的事件数据
- `config_file`: 自定义配置文件路径，默认为 `~/.baidu_tts_config.json`

#### 核心方法

**handle_tts_say(command)**
- **功能**：添加TTS播放任务到队列
- **参数**：
  ```python
  command = {
      "ts": 时间戳,
      "trace_id": "唯一标识",  # 必需
      "text": "要播放的文本",  # 必需，最大512字符
      "priority": 5,          # 必需，0-10，数值越高优先级越高
      "style": "default",      # 可选：default/news/cheerful/calm
      "interruptible": True    # 可选：是否可被中断
  }
  ```
- **返回**: `True`（成功）或 `False`（失败）

**handle_tts_stop(command)**
- **功能**：停止当前播放和清空可中断任务
- **参数**：
  ```python
  command = {
      "ts": 时间戳,
      "trace_id": "唯一标识",
      "source": "core",
      "reason": "preempt/user/error",
      "timeout_ms": 300,
      "instance": "实例标识"
  }
  ```

### 事件规范

模块通过回调函数发送以下类型的事件：

#### voice.tts_done（播放完成事件）
```python
{
    "ts": 时间戳,
    "trace_id": "对应任务的trace_id",
    "source": "voice",
    "ok": True/False,      # 播放是否成功
    "stopped": True/False   # 是否被中断
}
```

#### core.error_event（错误事件）
```python
{
    "ts": 时间戳,
    "trace_id": "相关任务ID",
    "source": "voice",
    "error_code": "错误代码",
    "message": "错误描述",
    "retry_after_ms": 5000  # 可选，重试间隔
}
```

## 4. 核心逻辑

### 工作流程
1. **任务接收**：通过`handle_tts_say`接收播放请求，验证格式后加入优先级队列
2. **语音合成**：工作线程从队列获取任务，调用百度API合成语音为MP3文件
3. **音频播放**：使用系统播放器播放生成的音频文件
4. **事件通知**：播放完成后发送`tts_done`事件

### 线程与并发处理
- **播放队列**：使用优先级队列管理多个播放任务
- **设备锁**：通过文件锁确保同一时间只有一个音频播放实例
- **线程安全**：播放线程与主线程通过队列通信，避免资源竞争
- **错误处理**：完善的异常捕获和错误事件上报机制

### 关键特性
- **优先级播放**：支持0-10级优先级，数值越高优先级越高
- **播放中断**：支持中断当前播放和清空可中断任务
- **跨平台兼容**：Windows/macOS/Linux多平台支持
- **临时文件管理**：音频文件保留在系统临时目录，可能需要core清理