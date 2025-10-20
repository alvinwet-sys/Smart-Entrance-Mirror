# 智能镜子系统 - 核心控制模块

## 1. 系统架构与实现

### 1.1 模块组成
```
core/
├── bus.py          # 事件总线：系统的"神经系统"，负责所有模块间的通信
├── config.py       # 配置管理：系统参数和阈值控制
├── healthcheck.py  # 健康检测：系统状态监控
├── main.py         # 主程序：系统启动和初始化
├── router.py       # 事件路由：核心业务逻辑处理
└── validator.py    # 数据验证：确保数据交换的正确性
```

### 1.2 核心功能实现

#### 1.2.1 事件总线系统 (bus.py)
```python
class EventBus:
    """
    实现：
    1. 异步事件队列 (asyncio.PriorityQueue)
    2. 发布-订阅模式
    3. 事件优先级 (critical=0, high=1, normal=5, low=9)
    4. 通配符订阅 (如 "core.*")
    5. JSON Schema 数据验证
    
    主要方法：
    - publish(type, data) -> bool  # 发布事件，返回是否成功
    - subscribe(type, handler)     # 订阅事件
    - unsubscribe(type, handler)   # 取消订阅
    """
```

#### 1.2.2 事件路由处理 (router.py)
```python
class Router:
    """
    功能：
    1. 维护系统状态
       - current_face: 当前识别的用户
       - last_asr: 最近的语音输入
       
    2. 处理核心事件
       - on_face: 处理人脸识别结果，触发欢迎语
       - on_asr_text: 处理语音输入，发起LLM请求
       - on_llm_decision: 处理LLM响应，控制TTS播报
       - on_tts_done: 处理TTS完成事件
    """
```

#### 1.2.3 配置管理 (config.py)
```python
class Config:
    """
    配置项：
    1. 阈值控制
       - face_confidence: 0.8  # 人脸识别置信度
       - asr_confidence: 0.7   # 语音识别置信度
       - wakeup_confidence: 0.6 # 唤醒检测置信度
       
    2. 系统限制
       - vision_max: 100  # 视觉模块队列长度
       - llm_max: 50     # LLM模块队列长度
    """
```

## 2. 工作流程详解

### 2.1 系统启动流程
```python
async def main():
    1. 加载配置 (load_cfg())
    2. 初始化事件总线 (EventBus())
    3. 创建路由器 (Router())
    4. 注册事件处理器
    5. 启动健康检查服务 (healthcheck)
    6. 启动事件循环 (bus.start())
```

### 2.2 人脸识别处理流程
```python
1. 视觉模块识别到人脸
2. 发布 core.face_id_resolved 事件
   {
       "ts": 1234567890,
       "trace_id": "uuid",
       "source": "vision",
       "keyword": "张三",      # 识别出的身份
       "confidence": 0.95
   }
   
3. Router.on_face 处理：
   - 检查置信度 (> 0.8)
   - 比对当前用户
   - 发送欢迎语 TTS
```

### 2.3 语音交互处理流程
```python
1. 收到语音识别文本 (voice.asr_text)
   {
       "text": "今天天气怎么样",
       "confidence": 0.85,
       "lang": "zh-CN"
   }
   
2. Router.on_asr_text 处理：
   - 验证置信度
   - 保存上下文
   - 发送决策请求
   
3. 收到 LLM 决策 (llm.decision_ready)
   {
       "reply_text": "今天天气晴朗",
       "priority": 7,
       "actions": ["update_display"]
   }
   
4. Router.on_llm_decision 处理：
   - 发送 TTS 播报
   - 执行额外动作
```

## 3. 数据验证机制

### 3.1 事件数据验证 (validator.py)
```python
class EventValidator:
    """
    验证内容：
    1. 必需字段完整性
    2. 字段类型正确性
    3. 取值范围合法性
    4. 枚举值有效性
    
    示例：
    - 时间戳必需且为数字
    - 置信度在 0-1 之间
    - 文本长度在指定范围内
    """
```

### 3.2 错误处理机制
```python
1. 数据验证错误
   - 生成错误事件
   - 记录详细日志
   
2. 处理器异常
   - 捕获异常
   - 发布错误事件
   - 可选的重试机制
```

## 4. 运行与调试

### 4.1 环境准备
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置文件
configs/config.yaml 示例：
app:
  http_port: 8080
thresholds:
  face_confidence: 0.8
  asr_confidence: 0.7
```

### 4.2 启动系统
```bash
# 方式1：直接启动
python core/main.py

# 方式2：指定配置
set SMART_MIRROR_CONFIG=configs/custom.yaml
python core/main.py
```

### 4.3 监控与调试
```bash
# 1. 健康检查
curl http://localhost:8080/healthz
预期输出：{"status": "ok"}

# 2. 日志格式
时间 - 模块名 - 级别 - 消息
示例：2025-10-20 10:00:00 - core.bus - INFO - Event published
```

## 5. 开发指南

### 5.1 添加新事件类型
1. 在 `contracts/events.py` 添加 Schema 定义
2. 在 `SCHEMAS` 字典中注册
3. 实现相应的事件处理器
4. 在 `main.py` 中注册处理器

### 5.2 自定义处理器
```python
async def custom_handler(evt: dict):
    # 1. 验证数据
    if not validate_data(evt):
        return
        
    # 2. 处理逻辑
    result = process_data(evt)
    
    # 3. 发布结果
    await bus.publish("custom.result", result)
```

### 5.3 常见问题排查
1. 事件未被处理
   - 检查事件类型是否正确注册
   - 验证数据格式是否符合 Schema
   - 查看处理器是否正确订阅

2. 配置未生效
   - 确认配置文件路径
   - 检查配置格式
   - 验证配置加载代码

3. 性能问题
   - 监控事件队列长度
   - 检查处理器执行时间
   - 观察资源使用情况
