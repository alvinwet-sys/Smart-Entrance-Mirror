#!/usr/bin/env python3
"""
智能镜系统主程序 - V2 - 引入多线程与状态机设计
"""

import asyncio
import threading
import queue
import time
import logging
import uuid
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


    # def handle_audio_message(self, msg):
    #     msg_type = msg.get('type')
        
    #     if msg_type == 'wake_word_detected':
    #         logger.info("🎤 检测到唤醒词！中断当前所有活动。")
    #         # 不论当前在做什么，都立即打断
    #         self.cancel_timeout_timer()
    #         self.message_bus.send('router_to_tts', {'type': 'tts_stop'})
    #         self.message_bus.send('system_control', {'type': 'pause_vision'}) # 确保视觉暂停
    #         self.set_state(SystemState.LISTENING)
    #         # 可以播放一个提示音表示已唤醒
    #         self.message_bus.send('router_to_tts', {
    #             'type': 'play_sound', 'sound': 'awake.wav', 'trace_id': 'awake_prompt'
    #         })
    #         # 设置用户说话超时
    #         self.start_timeout_timer(15, self.reset_to_idle, "用户输入超时，返回空闲状态。")

    #     elif msg_type == 'user_command' and self.state == SystemState.LISTENING:
    #         self.cancel_timeout_timer()
    #         text = msg.get('text')
    #         if not text:
    #             logger.warning("ASR 结果为空，返回聆听状态")
    #             # 可选择重新进入聆听或直接超时
    #             self.start_timeout_timer(15, self.reset_to_idle, "用户输入超时，返回空闲状态。")
    #             return
            
    #         logger.info(f"🗣️ 收到用户命令: '{text}'")
    #         self.set_state(SystemState.PROCESSING)
            
    #         # (伪代码) 在这里可以加入规则库匹配
    #         # if is_simple_command(text):
    #         #     ...
    #         # else:
    #         # 发送到LLM
    #         self.message_bus.send('router_to_llm', {
    #             'type': 'query',
    #             'text': text,
    #             'user': self.current_user
    #         })
    #         # 设置LLM处理超时
    #         self.start_timeout_timer(30, self.reset_to_idle, "处理超时，返回空闲状态。")

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
# 在 main.py 中，修改 StateMachineRouter 类

    def handle_tts_callback(self, msg):
        trace_id = msg.get('trace_id')
        if msg.get('status') == 'finished':
            # --- 这是关键改动 ---
            if self.state == SystemState.GREETING and trace_id == 'face_greeting':
                logger.info("👋 问候语播放完毕，进入聆听状态等待用户输入...")
                self.set_state(SystemState.LISTENING)
                # 设置用户输入超时
                self.start_timeout_timer(30, self.reset_to_idle, "用户输入超时，返回空闲状态。")
                # 在终端打印提示，方便测试
                print("\n=============================================")
                print("🤖 请在下方输入您的问题 (按Enter发送):")
                print("=============================================")
                sys.stdout.flush() # 确保提示立即显示

            elif self.state == SystemState.RESPONDING and trace_id == 'llm_response':
                 logger.info("✅ LLM回答播放完毕，对话结束。")
                 self.reset_to_idle() # 回答完就结束


    def handle_audio_message(self, msg):
        msg_type = msg.get('type')

        # (唤醒词逻辑保持不变，虽然我们这次不用它)
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
            

        # 主要处理来自终端的用户命令
        elif msg_type == 'user_command' and self.state == SystemState.LISTENING:
            self.cancel_timeout_timer()
            text = msg.get('text')
            if not text:
                logger.warning("输入为空，请重新输入。")
                # 重新进入聆听
                self.start_timeout_timer(30, self.reset_to_idle, "用户输入超时，返回空闲状态。")
                return

            logger.info(f"🗣️  收到用户命令 (from Terminal): '{text}'")
            self.set_state(SystemState.PROCESSING)

            # 发送到LLM
            self.message_bus.send('router_to_llm', {
                'type': 'query',
                'text': text,
                'user': self.current_user,
                'trace_id': f"query_{int(time.time())}"
            })
            # 设置LLM处理超时
            self.start_timeout_timer(45, self.reset_to_idle, "LLM处理超时，返回空闲状态。")

    # def handle_tts_callback(self, msg):
    #     trace_id = msg.get('trace_id')
    #     if msg.get('status') == 'finished':
    #         if self.state == SystemState.GREETING and trace_id == 'face_greeting':
    #              self.reset_to_idle() # 问候完就结束
    #         elif self.state == SystemState.RESPONDING and trace_id == 'llm_response':
    #              self.reset_to_idle() # 回答完就结束
    #              # 或者进入多轮对话： self.set_state(SystemState.LISTENING)
    
    

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
            
# 在 main.py 中，添加这个新类
class TerminalInputHandler:
    """
    通过终端命令行模拟用户的语音输入，用于测试。
    """
    def __init__(self, message_bus, router):
        self.message_bus = message_bus
        self.router = router # 需要引用 router 来检查当前状态

    def start_processing(self):
        logger.info("⌨️  终端输入处理器已启动，用于测试。")
        while self.message_bus.is_running:
            try:
                # 只有在路由器处于聆听状态时才阻塞等待输入
                if self.router.state == SystemState.LISTENING:
                    # input() 会阻塞当前线程，这是我们想要的
                    user_text = input("> ")
                    
                    # 确保在发送消息时，状态仍然是LISTENING
                    if self.router.state == SystemState.LISTENING and user_text.strip():
                        # 模拟 ASR/Audio 模块发送用户命令
                        self.message_bus.send('audio_to_router', {
                            'type': 'user_command',
                            'text': user_text.strip()
                        })
                else:
                    # 如果不是聆听状态，就短暂休眠，避免CPU空转
                    time.sleep(0.5)
            except (EOFError, KeyboardInterrupt):
                # 用户按 Ctrl+D 或 Ctrl+C
                logger.info("终端输入处理器已停止。")
                break
            except Exception as e:
                logger.error(f"终端输入时发生错误: {e}")

# ----------------- 其他模块适配 -----------------

# 在 main.py 中，使用这个类替换 SimpleTTSWrapper
# (请确保 voice_llm/tts_module.py 路径已添加到 sys.path)
from voice_llm.tts_module import TTSModule # 导入您的模块

class SimpleTTSWrapper:
    """
    集成了 TTSModule 的TTS模块包装器
    """
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.tts_module = None

    def _on_tts_event_callback(self, event_data):
        """
        当 TTSModule 完成播放或发生错误时，此函数被调用。
        它将结果格式化并发送回路由器。
        """
        # 检查是否是播放完成的事件
        if event_data.get("source") == "voice" and "ok" in event_data:
            logger.info(f"🔊 TTS 回调: trace_id={event_data['trace_id']}, "
                        f"stopped={event_data.get('stopped', False)}")
            
            # 格式化为路由器期望的回调消息
            router_callback_msg = {
                'status': 'finished',
                'trace_id': event_data.get('trace_id')
            }
            self.message_bus.send('tts_callback_to_router', router_callback_msg)
        
        # 也可以在这里处理错误事件
        elif "error_code" in event_data:
             logger.error(f"❌ TTS 模块内部错误: {event_data['message']}")

    def start_processing(self):
        """
        作为 TTS 线程的入口函数。
        """
        logger.info("🔊 正在初始化TTS模块...")
        try:
            # 初始化您的 TTSModule，并传入我们的回调函数
            self.tts_module = TTSModule(event_callback=self._on_tts_event_callback)
            logger.info("🔊 TTS模块初始化完成，监听播报任务...")

            while self.message_bus.is_running:
                # 从路由器接收消息
                msg = self.message_bus.receive('router_to_tts')
                if msg is None:
                    continue

                msg_type = msg.get('type')
                if msg_type == 'tts_say':
                    # 将我们的消息格式转换为 TTSModule 期望的 schema 格式
                    tts_command = {
                        "ts": time.time(),
                        "trace_id": msg.get('trace_id', str(uuid.uuid4())),
                        "text": msg.get('text', ''),
                        "priority": msg.get('priority', 5),
                        "interruptible": True
                    }
                    self.tts_module.handle_tts_say(tts_command)
                
                elif msg_type == 'tts_stop':
                     # 创建一个符合 schema 的 stop 命令
                    stop_command = {
                        "ts": time.time(),
                        "trace_id": msg.get('trace_id', 'stop_request'),
                        "source": "core",
                        "reason": "preempt"
                    }
                    self.tts_module.handle_tts_stop(stop_command)

        except Exception as e:
            logger.error(f"❌ TTS模块启动或运行失败: {e}", exc_info=True)

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

# 在 main.py 中，使用这个类替换 SimpleVisionSystem
# (请确保 vision/face_reco.py 路径已添加到 sys.path)
from vision.face_reco import RealTimeFaceRecognition # 导入您的模块

class SimpleVisionSystem:
    """
    集成了 RealTimeFaceRecognition 的视觉模块包装器
    """
    def __init__(self, message_bus, config):
        self.message_bus = message_bus
        self.config = config
        self.recognizer = None

    def _on_face_detected_callback(self, event_type, event_data):
        """
        当 RealTimeFaceRecognition 识别到人脸时，此函数被调用。
        它将事件数据发送到消息总线。
        """
        # 我们只关心事件数据，将其发送到路由器
        # event_data 已经是您期望的格式 {'keyword': ..., 'confidence': ...}
        logger.info(f"👁️  视觉回调: 检测到 {event_data.get('keyword')}")
        self.message_bus.send('vision_to_router', event_data)

    def start_processing(self):
        """
        作为 Vision 线程的入口函数。
        """
        logger.info("👁️  正在初始化视觉模块...")
        try:
            self.recognizer = RealTimeFaceRecognition(
                model_path=self.config.vision['model_path'],
                gallery_dir=self.config.vision['gallery_dir']
            )
            # 设置回调函数，将识别结果连接到我们的消息总线
            self.recognizer.set_direct_callback(self._on_face_detected_callback)
            
            logger.info("👁️  视觉模块初始化完成，启动处理循环...")
            
            # RealTimeFaceRecognition 内部管理着自己的处理线程和asyncio循环
            # 我们需要在一个新的事件循环中启动它
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 启动识别器（这将启动其内部的处理线程）
            loop.run_until_complete(self.recognizer.start())
            
            # 让当前线程保持存活，以接收暂停/恢复指令
            while self.message_bus.is_running:
                time.sleep(1)

        except Exception as e:
            logger.error(f"❌ 视觉模块启动失败: {e}", exc_info=True)
        finally:
            if self.recognizer and self.recognizer.is_running:
                # 确保在退出时停止
                loop.run_until_complete(self.recognizer.stop())

    def pause(self):
        """外部接口，用于暂停视觉处理"""
        if self.recognizer:
            self.recognizer.pause()

    def resume(self):
        """外部接口，用于恢复视觉处理"""
        if self.recognizer:
            self.recognizer.resume()
            

# 在 main.py 的模块定义区域添加这个新类
# 确保 llm_interface.py 所在的路径已添加到 sys.path
from voice_llm.llm_worker.llm_interface import LLMInterface # 导入您的模块

class SimpleLLMWrapper:
    """
    集成了 LLMInterface 的LLM模块包装器
    """
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.llm_interface = None

    def start_processing(self):
        """
        作为 LLM 线程的入口函数。
        """
        logger.info("🧠 正在初始化LLM模块...")
        try:
            # 初始化您的 LLMInterface 实例
            self.llm_interface = LLMInterface()
            logger.info("🧠 LLM模块初始化完成，等待决策请求...")

            while self.message_bus.is_running:
                # 从路由器接收消息
                msg = self.message_bus.receive('router_to_llm')
                if msg is None:
                    continue

                if msg.get('type') == 'query':
                    # 1. 将我们的内部消息格式转换为 LLMInterface 期望的事件格式
                    decision_request_event = {
                        "ts": time.time(),
                        "trace_id": msg.get('trace_id', str(uuid.uuid4())),
                        "source": "core",
                        "query": msg.get('text', ''),
                        "context": {
                            "identity": msg.get('user', 'Unknown')
                        }
                    }

                    # 2. 调用 LLM 核心处理方法
                    result = self.llm_interface.handle_decision_request(decision_request_event)

                    # 3. 将返回结果转换为我们的内部消息格式，并发送回路由器
                    if 'reply_text' in result:
                        response_msg = {
                            'type': 'llm_response',
                            'text': result['reply_text'],
                            'trace_id': result.get('trace_id')
                        }
                        self.message_bus.send('llm_to_router', response_msg)
                    elif 'error_code' in result:
                        logger.error(f"❌ LLM处理错误: {result['error_code']} - {result['message']}")
                        # 也可以将错误信息包装后发回路由器，让TTS播报错误
                        error_response_msg = {
                            'type': 'llm_response',
                            'text': "抱歉，我好像遇到了一点麻烦。",
                            'trace_id': result.get('trace_id')
                        }
                        self.message_bus.send('llm_to_router', error_response_msg)

        except Exception as e:
            logger.error(f"❌ LLM模块启动或运行失败: {e}", exc_info=True)
        

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
class AppConfig:
    vision = {
        'model_path': r"C:/Users/17321/Desktop/core_run/core/model/w600k_r50.onnx",
        'gallery_dir': r"C:/Users/17321/Desktop/core_run/core/vision/gallery_dataset"
    }
    # 未来可以添加 TTS, LLM 等配置
    tts = {}

class SmartMirrorApp:
    def __init__(self):
        self.message_bus = SimpleMessageBus()
        self.threads = []
        self.config = AppConfig()
        
        # 初始化所有模块
        self.vision = SimpleVisionSystem(self.message_bus,self.config)
        self.audio = AudioSystem(self.message_bus)
        self.router = StateMachineRouter(self.message_bus)
        self.tts = SimpleTTSWrapper(self.message_bus)
        self.llm = SimpleLLMWrapper(self.message_bus)
        self.terminal_input = TerminalInputHandler(self.message_bus,self.router) #用于测试LLM
        self.controller = SystemController(self.message_bus, self.vision)

        self.modules = {
            "Router": self.router,
            "Vision": self.vision,
            "Audio": self.audio,
            "TTS": self.tts,
            "LLM": self.llm,
            "Controller": self.controller,
            "TerminalInput":self.terminal_input
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