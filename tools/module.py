#!/usr/bin/env python3
"""
智能镜系统主程序 - V2 - 引入多线程与状态机设计
"""

import asyncio
import threading
import queue
import time
import logging
import sys
import os

# 添加模块路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 状态定义 ---
class SystemState:
    IDLE = "IDLE"          # 空闲，等待人脸或唤醒
    GREETING = "GREETING"      # 正在识别人脸并问候
    LISTENING = "LISTENING"    # 已唤醒，等待用户语音命令
    PROCESSING = "PROCESSING"  # 正在处理用户命令（LLM/规则）
    RESPONDING = "RESPONDING"  # 正在通过TTS播报回复

class SimpleMessageBus:
    """简化的消息总线，使用队列进行模块间通信"""
    def __init__(self):
        self.queues = {
            'vision_to_router': queue.Queue(),
            'audio_to_router': queue.Queue(),      # 新增：音频输入到路由
            'router_to_tts': queue.Queue(),
            'router_to_llm': queue.Queue(),
            'llm_to_router': queue.Queue(),        # 修改：LLM回复给路由
            'system_control': queue.Queue(),
            'tts_callback_to_router': queue.Queue() # 新增：TTS回调到路由
        }
        self.is_running = True

    def send(self, queue_name, message):
        if queue_name in self.queues and self.is_running:
            self.queues[queue_name].put(message)
            # logger.info(f"📨 -> {queue_name}: {message.get('type', 'unknown')}")
        else:
            logger.error(f"❌ 无法发送消息，队列不存在或已停止: {queue_name}")

    def receive(self, queue_name, timeout=1.0):
        if queue_name in self.queues and self.is_running:
            try:
                return self.queues[queue_name].get(timeout=timeout)
            except queue.Empty:
                return None
        return None

    def stop(self):
        self.is_running = False
        for q in self.queues.values():
            # 放入一个哨兵值来确保阻塞的get()可以退出
            q.put(None)

# ----------------- 重构后的核心模块 -----------------

class StateMachineRouter:
    """
    核心路由器，作为状态机来管理系统流程
    """
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.state = SystemState.IDLE
        self.current_user = None
        self.last_face_time = 0
        self.conversation_timeout_timer = None

    def set_state(self, new_state):
        if self.state != new_state:
            logger.info(f"🔄 状态转换: {self.state} -> {new_state}")
            self.state = new_state

    def start_processing(self):
        logger.info("🤖 路由器（状态机）启动...")
        self.set_state(SystemState.IDLE)
        # 初始时恢复视觉处理
        self.message_bus.send('system_control', {'type': 'resume_vision'})
        
        while self.message_bus.is_running:
            # 优先处理高优先级消息，如语音唤醒
            audio_msg = self.message_bus.receive('audio_to_router', timeout=0.05)
            if audio_msg:
                self.handle_audio_message(audio_msg)
                continue

            # 按顺序处理其他消息
            vision_msg = self.message_bus.receive('vision_to_router', timeout=0.05)
            if vision_msg:
                self.handle_vision_message(vision_msg)
                continue
            
            tts_callback = self.message_bus.receive('tts_callback_to_router', timeout=0.05)
            if tts_callback:
                self.handle_tts_callback(tts_callback)
                continue
            
            llm_response = self.message_bus.receive('llm_to_router', timeout=0.05)
            if llm_response:
                self.handle_llm_response(llm_response)
                continue


    def handle_vision_message(self, msg):
        # 只有在空闲状态下才响应人脸识别
        if self.state == SystemState.IDLE:
            identity = msg.get('keyword')
            confidence = msg.get('confidence', 0)
            current_time = time.time()

            if (confidence > 0.6 and identity != "stranger" and
                    current_time - self.last_face_time > 15.0): # 15秒冷却
                
                self.last_face_time = current_time
                self.current_user = identity
                self.set_state(SystemState.GREETING)

                # 暂停视觉以避免干扰
                self.message_bus.send('system_control', {'type': 'pause_vision'})
                
                greeting = f"你好，{self.current_user}！今天看起来不错。"
                self.message_bus.send('router_to_tts', {
                    'type': 'tts_say',
                    'text': greeting,
                    'trace_id': 'face_greeting'
                })
                # 设置一个定时器，问候后若无操作则超时返回IDLE
                self.start_timeout_timer(10, self.reset_to_idle)


    def handle_audio_message(self, msg):
        msg_type = msg.get('type')
        
        if msg_type == 'wake_word_detected':
            logger.info("🎤 检测到唤醒词！中断当前所有活动。")
            # 不论当前在做什么，都立即打断
            self.cancel_timeout_timer()
            self.message_bus.send('router_to_tts', {'type': 'tts_stop'})
            self.message_bus.send('system_control', {'type': 'pause_vision'}) # 确保视觉暂停
            self.set_state(SystemState.LISTENING)
            # 可以播放一个提示音表示已唤醒
            self.message_bus.send('router_to_tts', {
                'type': 'play_sound', 'sound': 'awake.wav', 'trace_id': 'awake_prompt'
            })
            # 设置用户说话超时
            self.start_timeout_timer(15, self.reset_to_idle, "用户输入超时，返回空闲状态。")

        elif msg_type == 'user_command' and self.state == SystemState.LISTENING:
            self.cancel_timeout_timer()
            text = msg.get('text')
            if not text:
                logger.warning("ASR 结果为空，返回聆听状态")
                # 可选择重新进入聆听或直接超时
                self.start_timeout_timer(15, self.reset_to_idle, "用户输入超时，返回空闲状态。")
                return
            
            logger.info(f"🗣️ 收到用户命令: '{text}'")
            self.set_state(SystemState.PROCESSING)
            
            # (伪代码) 在这里可以加入规则库匹配
            # if is_simple_command(text):
            #     ...
            # else:
            # 发送到LLM
            self.message_bus.send('router_to_llm', {
                'type': 'query',
                'text': text,
                'user': self.current_user
            })
            # 设置LLM处理超时
            self.start_timeout_timer(30, self.reset_to_idle, "处理超时，返回空闲状态。")

    def handle_llm_response(self, msg):
        if self.state == SystemState.PROCESSING:
            self.cancel_timeout_timer()
            response_text = msg.get('text')
            logger.info(f"🧠 LLM 回复: '{response_text}'")
            self.set_state(SystemState.RESPONDING)
            self.message_bus.send('router_to_tts', {
                'type': 'tts_say',
                'text': response_text,
                'trace_id': 'llm_response'
            })
    
    def handle_tts_callback(self, msg):
        trace_id = msg.get('trace_id')
        if msg.get('status') == 'finished':
            if self.state == SystemState.GREETING and trace_id == 'face_greeting':
                 self.reset_to_idle() # 问候完就结束
            elif self.state == SystemState.RESPONDING and trace_id == 'llm_response':
                 self.reset_to_idle() # 回答完就结束
                 # 或者进入多轮对话： self.set_state(SystemState.LISTENING)

    def start_timeout_timer(self, duration, callback, message="操作超时。"):
        self.cancel_timeout_timer()
        self.conversation_timeout_timer = threading.Timer(duration, lambda: self.timeout_handler(callback, message))
        self.conversation_timeout_timer.start()

    def cancel_timeout_timer(self):
        if self.conversation_timeout_timer:
            self.conversation_timeout_timer.cancel()
            self.conversation_timeout_timer = None
            
    def timeout_handler(self, callback, message):
        logger.warning(f"⏰ {message}")
        callback()

    def reset_to_idle(self):
        self.cancel_timeout_timer()
        if self.state != SystemState.IDLE:
            self.set_state(SystemState.IDLE)
            self.current_user = None
            # 恢复视觉系统，准备下一次交互
            self.message_bus.send('system_control', {'type': 'resume_vision'})
            logger.info("系统已重置为空闲状态。")

# ----------------- 其他模块适配 -----------------

class SimpleTTSWrapper:
    """TTS包装器，增加中断功能"""
    def __init__(self, message_bus):
        self.message_bus = message_bus
        # self.tts = None # 假设这是你的TTS实例
        from voice_llm.tts_module import TTSModule
        self.tts = TTSModule(event_callback=self._on_tts_event)

    def start_processing(self):
        logger.info("🔊 TTS 模块启动...")
        while self.message_bus.is_running:
            msg = self.message_bus.receive('router_to_tts')
            if msg is None: continue

            msg_type = msg.get('type')
            if msg_type == 'tts_say':
                logger.info(f"TTS 准备播报: {msg.get('text')[:30]}...")
                # 伪代码: 调用真实TTS播放
                # self.tts.handle_tts_say(text=msg.get('text'), trace_id=msg.get('trace_id'))
                # 模拟播放和回调
                def simulate_playback(m):
                    time.sleep(3) # 模拟播放耗时
                    self.message_bus.send('tts_callback_to_router', {
                        'status': 'finished',
                        'trace_id': m.get('trace_id')
                    })
                threading.Thread(target=simulate_playback, args=(msg,)).start()

            elif msg_type == 'tts_stop':
                logger.info("TTS 收到中断指令！")
                # 伪代码: 调用真实TTS停止
                # self.tts.stop()

            elif msg_type == 'play_sound':
                logger.info(f"TTS 播放提示音: {msg.get('sound')}")
                # 伪代码: 播放WAV文件
                # self.tts.play(file=msg.get('sound'))

class AudioSystem:
    """新的音频处理模块"""
    def __init__(self, message_bus):
        self.message_bus = message_bus
    
    def start_processing(self):
        logger.info("🎤 音频模块启动，等待唤醒词...")
        # 这是一个模拟器，实际应使用真实音频流和唤醒词引擎
        def mock_input():
            time.sleep(8)
            logger.info("--- (模拟用户说出唤醒词) ---")
            self.message_bus.send('audio_to_router', {'type': 'wake_word_detected'})
            
            time.sleep(2)
            logger.info("--- (模拟用户说出 '今天天气怎么样') ---")
            self.message_bus.send('audio_to_router', {
                'type': 'user_command', 'text': '今天天气怎么样'
            })

        threading.Thread(target=mock_input, daemon=True).start()
        
        while self.message_bus.is_running:
            # 在实际应用中，这里是音频流处理循环
            # audio_chunk = stream.read()
            # wake_word_detected = porcupine.process(audio_chunk)
            # if wake_word_detected: ...
            time.sleep(1)

class SimpleVisionSystem:
    """简化的视觉系统 - 支持暂停/恢复 (与您原版类似)"""
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.is_paused = True # 初始时暂停

    def start_processing(self):
        logger.info("👁️ 视觉模块启动...")
        # 模拟人脸识别
        def mock_recognition():
            time.sleep(5)
            if not self.is_paused:
                logger.info("--- (模拟识别到人脸 'John') ---")
                self.message_bus.send('vision_to_router', {
                    'type': 'face_detected',
                    'keyword': 'John',
                    'confidence': 0.95
                })
        
        while self.message_bus.is_running:
            if not self.is_paused:
                # 实际的人脸识别代码在这里循环
                # 这里用一个延时和一次性任务模拟
                threading.Thread(target=mock_recognition).start()
                time.sleep(10) # 降低频率
            else:
                time.sleep(1) # 暂停时低功耗等待

    def pause(self):
        if not self.is_paused:
            self.is_paused = True
            logger.info("⏸️ 视觉系统已暂停")

    def resume(self):
        if self.is_paused:
            self.is_paused = False
            logger.info("▶️ 视觉系统已恢复")

# LLM 和 SystemController 可以类似地简化和适配
class SimpleLLMWrapper:
    def __init__(self, message_bus):
        self.message_bus = message_bus
    def start_processing(self):
        logger.info("🧠 LLM 模块启动...")
        while self.message_bus.is_running:
            msg = self.message_bus.receive('router_to_llm')
            if msg:
                logger.info(f"LLM 正在处理: {msg.get('text')}")
                time.sleep(2) # 模拟处理延迟
                self.message_bus.send('llm_to_router', {
                    'type': 'llm_response',
                    'text': '今天天气晴朗，适合出门散步。'
                })

class SystemController:
    def __init__(self, message_bus, vision_system):
        self.message_bus = message_bus
        self.vision_system = vision_system
    def start_processing(self):
        logger.info("🎛️ 系统控制器启动...")
        while self.message_bus.is_running:
            msg = self.message_bus.receive('system_control')
            if msg:
                if msg.get('type') == 'pause_vision':
                    self.vision_system.pause()
                elif msg.get('type') == 'resume_vision':
                    self.vision_system.resume()


# ----------------- 主程序 -----------------
class SmartMirrorApp:
    def __init__(self):
        self.message_bus = SimpleMessageBus()
        self.threads = []
        
        # 初始化所有模块
        self.vision = SimpleVisionSystem(self.message_bus)
        self.audio = AudioSystem(self.message_bus)
        self.router = StateMachineRouter(self.message_bus)
        self.tts = SimpleTTSWrapper(self.message_bus)
        self.llm = SimpleLLMWrapper(self.message_bus)
        self.controller = SystemController(self.message_bus, self.vision)

        self.modules = {
            "Router": self.router,
            "Vision": self.vision,
            "Audio": self.audio,
            "TTS": self.tts,
            "LLM": self.llm,
            "Controller": self.controller
        }

    def start(self):
        logger.info("🚀 启动智能镜系统...")
        
        for name, module in self.modules.items():
            thread = threading.Thread(target=module.start_processing, name=name, daemon=True)
            self.threads.append(thread)
            thread.start()
        
        logger.info("✅ 所有模块线程已启动。按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("⏹️ 收到中断信号，开始关闭系统...")
            self.stop()
    
    def stop(self):
        self.message_bus.stop()
        logger.info("🛑 等待所有线程退出...")
        # 主线程等待所有子线程结束
        # 由于设置了daemon=True, 主线程退出时它们也会被强制结束
        # 但一个更优雅的方式是让每个循环都检查self.message_bus.is_running
        time.sleep(1)
        logger.info("系统已关闭。")

if __name__ == "__main__":
    app = SmartMirrorApp()
    app.start()