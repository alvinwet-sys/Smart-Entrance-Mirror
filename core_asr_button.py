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
import numpy as np
import warnings

# 抑制各种警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*trust_remote_code.*", category=UserWarning)

# 设置环境变量来抑制某些第三方库的警告
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# 添加模块路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 集成 ASR 模块 ---
try:
    # 假设 asr_service.py 位于 voice_llm/ 目录下
    from voice_llm.asr_service import FunASRWrapper, SimpleVAD, CONFIG as ASR_CONFIG, gen_trace_id
    import pyaudio
    ASR_MODULE_LOADED = True
except ImportError as e:
    logger.error(f"❌ 无法导入ASR模块，请确保 asr_service.py 路径正确: {e}")
    FunASRWrapper = None
    SimpleVAD = None
    pyaudio = None
    ASR_MODULE_LOADED = False

# --- 集成按键监听 ---
try:
    import keyboard
    KEYBOARD_MODULE_LOADED = True
except ImportError as e:
    logger.warning(f"⚠️ 按键监听模块未安装: {e}")
    KEYBOARD_MODULE_LOADED = False

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

# ----------------- 按键监听模块 -----------------

class KeyboardWakeupListener:
    """
    按键监听器 - 唤醒ASR系统
    支持空格键和回车键唤醒
    """
    def __init__(self, message_bus, config=None):
        self.message_bus = message_bus
        self.config = config
        self.enabled = True
        self.wakeup_keys = ['space', 'enter']  # 默认唤醒键
        
        if config and hasattr(config, 'keyboard'):
            self.enabled = config.keyboard.get('enable_keyboard_wakeup', True)
            self.wakeup_keys = config.keyboard.get('wakeup_keys', ['space', 'enter'])
        
        logger.info(f"⌨️ 按键唤醒: {'启用' if self.enabled else '禁用'}, 唤醒键: {self.wakeup_keys}")

    def start_processing(self):
        """按键监听线程入口"""
        if not self.enabled:
            logger.info("⌨️ 按键唤醒已禁用，跳过按键监听")
            while self.message_bus.is_running:
                time.sleep(1)
            return
            
        if not KEYBOARD_MODULE_LOADED:
            logger.error("❌ keyboard模块未安装，无法启动按键监听")
            logger.error("💡 请运行: pip install keyboard")
            while self.message_bus.is_running:
                time.sleep(1)
            return

        logger.info("⌨️ 启动按键唤醒监听器...")
        
        try:
            def on_key_event(event):
                if event.event_type == keyboard.KEY_DOWN and event.name in self.wakeup_keys:
                    logger.info(f"⌨️ 检测到唤醒按键: {event.name}")
                    self.message_bus.send('audio_to_router', {
                        'type': 'keyboard_wakeup',
                        'key': event.name,
                        'confidence': 1.0
                    })
            
            # 注册全局按键监听
            keyboard.hook(on_key_event)
            
            logger.info(f"✅ 按键唤醒监听器已启动，监听按键: {self.wakeup_keys}")
            
            # 保持线程存活
            while self.message_bus.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ 按键监听器启动失败: {e}", exc_info=True)
        finally:
            try:
                if KEYBOARD_MODULE_LOADED:
                    keyboard.unhook_all()
                logger.info("⌨️ 按键监听器已停止")
            except:
                pass

# ----------------- 重构后的核心模块 -----------------

class StateMachineRouter:
    """
    核心路由器，作为状态机来管理系统流程
    """
    def __init__(self, message_bus, vision_system=None, config=None):
        self.message_bus = message_bus
        self.vision_system = vision_system  # 直接引用视觉系统
        self.config = config  # 新增：配置参数
        self.state = SystemState.IDLE
        self.current_user = None
        self.last_face_time = 0
        self.conversation_timeout_timer = None
        
        # 拼音标签到中文人名的映射表
        self.name_mapping = {
            # 基于目录名的映射
            'zhao_tianyu': '赵天宇',
            'zhaotianyu': '赵天宇',
            'wang_jianzhao': '王建昭',
            'wangjianzhao': '王建昭', 
            'wang_lidong': '王立栋',
            'wanglidong': '王立栋',
            'fang_haidie': '方海蝶',
            'fanghaidie': '方海蝶',
            'sang_jiayi': '桑嘉一',
            'sangjiayi': '桑嘉一',
            'ye_weirui': '叶维瑞',
            'yeweirui': '叶维瑞',
            'zhang_chi': '张驰',
            'zhangchi': '张驰',
            'zhu_zirui': '朱子睿',
            'zhuzirui': '朱子睿',
            # 可以继续添加更多映射
        }

    def set_state(self, new_state):
        if self.state != new_state:
            logger.info(f"🔄 状态转换: {self.state} -> {new_state}")
            self.state = new_state

    def get_chinese_name(self, pinyin_label):
        """
        将拼音标签转换为中文人名，用于TTS播报
        """
        if not pinyin_label:
            return "朋友"  # 默认称呼
        
        # 转为小写进行匹配
        pinyin_lower = pinyin_label.lower()
        chinese_name = self.name_mapping.get(pinyin_lower, pinyin_label)
        
        if chinese_name != pinyin_label:
            logger.info(f"📝 人名映射: '{pinyin_label}' -> '{chinese_name}'")
        else:
            logger.warning(f"⚠️ 未找到 '{pinyin_label}' 的中文映射，使用原始标签")
        
        return chinese_name

    def start_processing(self):
        logger.info("🤖 路由器（状态机）启动...")
        self.set_state(SystemState.IDLE)
        
        # 等待视觉系统初始化完成，然后恢复
        if self.vision_system:
            # 在单独的线程中异步等待并恢复视觉模块
            def wait_and_resume():
                max_wait_time = 30  # 最大等待30秒
                wait_count = 0
                logger.info("🔍 开始等待视觉模块初始化...")
                while not hasattr(self.vision_system, 'is_ready') or not self.vision_system.is_ready:
                    time.sleep(0.5)
                    wait_count += 1
                    if wait_count % 4 == 0:  # 每2秒输出一次
                        logger.info(f"🔍 等待视觉模块初始化... ({wait_count * 0.5:.1f}s)")
                    if wait_count > max_wait_time * 2:  # 0.5秒 * 60 = 30秒
                        logger.warning("⚠️ 等待视觉模块超时，强制启动")
                        break
                
                if self.message_bus.is_running:
                    logger.info("🔍 视觉模块已准备就绪，开始恢复...")
                    self.vision_system.resume()
                    logger.info("🎛️ 路由器已恢复视觉模块")
            
            threading.Thread(target=wait_and_resume, daemon=True).start()
        
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
                    current_time - self.last_face_time > 30.0): # 增加到30秒冷却，防止重复触发
                
                self.last_face_time = current_time
                self.current_user = identity
                
                # 立即暂停视觉模块 - 增加调试信息
                logger.info(f"🔍 准备暂停视觉模块，vision_system={self.vision_system is not None}")
                if self.vision_system:
                    logger.info(f"🔍 视觉模块状态: recognizer={self.vision_system.recognizer is not None}")
                    self.vision_system.pause()
                    logger.info("🎛️ 已立即暂停视觉模块")
                else:
                    logger.error("❌ vision_system 为 None，无法暂停")
                
                self.set_state(SystemState.GREETING)
                
                # 合并问候语和聆听提示 - 使用中文人名
                chinese_name = self.get_chinese_name(self.current_user)
                greeting = f"你好，{chinese_name}！今天看起来不错。有什么可以帮您的吗？"
                self.message_bus.send('router_to_tts', {
                    'type': 'tts_say',
                    'text': greeting,
                    'trace_id': 'combined_greeting'
                })
                # 设置一个定时器，问候后若无操作则超时返回IDLE
                # 确保超时时间大于预估的TTS播放时长
                estimated_greeting_duration = self.estimate_playback_duration(greeting)
                timeout_duration = max(20, estimated_greeting_duration + 10)  # 至少20秒，或播放时长+10秒缓冲
                logger.info(f"⏰ 设置问候超时: {timeout_duration:.1f}秒 (TTS预估: {estimated_greeting_duration:.1f}秒)")
                self.start_timeout_timer(timeout_duration, self.reset_to_idle)
            else:
                if confidence <= 0.6:
                    logger.debug(f"👁️ 人脸置信度过低: {confidence:.3f}")
                elif identity == "stranger":
                    logger.debug(f"👁️ 检测到陌生人，跳过")
                else:
                    remaining_cooldown = 30.0 - (current_time - self.last_face_time)
                    logger.debug(f"👁️ 人脸识别冷却中，剩余: {remaining_cooldown:.1f}秒")
        else:
            # 非IDLE状态下收到人脸识别消息，记录但不处理
            identity = msg.get('keyword')
            confidence = msg.get('confidence', 0)
            logger.info(f"👁️ 当前状态为 {self.state}，忽略人脸识别: {identity} (置信度: {confidence:.3f})")


    def handle_audio_message(self, msg):
        msg_type = msg.get('type')
        
        if msg_type == 'keyboard_wakeup':
            key_name = msg.get('key', '按键')
            logger.info(f"⚨️ 即时唤醒: {key_name} | 当前状态: {self.state} -> LISTENING")
            
            # 🚨 立即打断模式：不论当前状态，强制进入LISTENING
            self.cancel_timeout_timer()
            
            # 🔥 关键修复：先停止TTS，等待一小段时间确保停止生效
            self.message_bus.send('router_to_tts', {'type': 'tts_stop'})
            
            # 立即暂停视觉模块
            if self.vision_system:
                self.vision_system.pause()
                logger.info("🎛️ 视觉模块已暂停")
            
            # 🔥 重要：立即切换状态，避免接收到延迟的TTS回调
            self.set_state(SystemState.LISTENING)
            
            # 🎯 延迟发送"请说"，确保TTS停止命令先执行
            def send_delayed_feedback():
                # 再次检查状态，确保仍在LISTENING（防止状态被其他事件改变）
                if self.state == SystemState.LISTENING:
                    feedback_text = "请说"
                    self.message_bus.send('router_to_tts', {
                        'type': 'tts_say',
                        'text': feedback_text,
                        'trace_id': 'instant_wakeup_feedback'
                    })
                    logger.info("🎤 已发送唤醒反馈: 请说")
                else:
                    logger.info(f"🎤 跳过唤醒反馈 - 状态已变为: {self.state}")
            
            # 延迟200ms发送反馈，确保TTS停止生效
            feedback_timer = threading.Timer(0.2, send_delayed_feedback)
            feedback_timer.start()
            
            # 用户输入超时保护（较长时间，避免误触发）
            self.start_timeout_timer(15, self.reset_to_idle, "语音输入超时")

        elif msg_type == 'user_command' and self.state == SystemState.LISTENING:
            self.cancel_timeout_timer()
            text = msg.get('text', '').strip()
            
            # 简化的ASR过滤：只过滤明显无效的输入
            if not text or len(text.strip()) < 1:
                logger.warning("🔇 ASR结果无效，返回空闲状态")
                self.reset_to_idle()
                return
            
            logger.info(f"🗣️ 用户语音: '{text}'")
            self.set_state(SystemState.PROCESSING)
            
            # 发送到LLM处理
            self.message_bus.send('router_to_llm', {
                'type': 'query',
                'text': text,
                'user': self.current_user
            })
            
            # LLM处理超时保护
            self.start_timeout_timer(25, self.reset_to_idle, "LLM处理超时")

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
        """
        处理TTS播放完成的回调 - 改进版，增加状态验证
        采用"单次对话"模式，每次回复后立即返回IDLE，等待新的唤醒
        """
        trace_id = msg.get('trace_id')
        if msg.get('status') != 'finished':
            return

        logger.info(f"🔊 收到TTS回调: {trace_id}, 当前状态: {self.state}")

        # 🔥 关键修复：检查回调是否过时
        # 如果系统已经被按键唤醒进入LISTENING状态，忽略之前的LLM回复回调
        if trace_id == 'llm_response' and self.state == SystemState.LISTENING:
            logger.info(f"🔊 ⏹️ 忽略过时的LLM回复回调 (系统已被重新唤醒)")
            return
        
        # 如果系统在IDLE状态收到任何回调，可能是延迟回调，也忽略
        if self.state == SystemState.IDLE and trace_id in ['llm_response', 'combined_greeting']:
            logger.info(f"🔊 ⏹️ 忽略IDLE状态下的延迟回调: {trace_id}")
            return

        # 取消现有的超时定时器
        self.cancel_timeout_timer()

        # 1. 问候语播放完毕 -> 直接返回空闲状态
        if self.state == SystemState.GREETING and trace_id == 'combined_greeting':
            logger.info("👋 问候播放完毕，返回空闲状态")
            self.reset_to_idle()
            return
        
        # 2. 即时唤醒反馈播放完毕 -> 保持聆听状态，等待用户命令  
        elif self.state == SystemState.LISTENING and trace_id in ['keyboard_wakeup_feedback', 'instant_wakeup_feedback']:
            logger.info("⚡ 唤醒反馈完成，等待语音输入...")
            # 保持LISTENING状态，等待用户语音输入
            return

        # 3. LLM回答播放完毕 -> 立即返回空闲状态（单次对话模式）
        elif self.state == SystemState.RESPONDING and trace_id == 'llm_response':
            logger.info("✅ LLM回复播放完毕，返回空闲状态（单次对话完成）")
            self.reset_to_idle()
            return
        
        # 4. 其他情况 - 可能是状态不匹配的过时回调
        else:
            logger.info(f"🔊 ⚠️ 状态不匹配的TTS回调: trace_id={trace_id}, 状态={self.state}, 忽略处理")

    def estimate_playback_duration(self, text):
        """
        估算TTS播放时长
        基于文本长度和平均语速进行估算 - 支持长文本
        """
        if not text:
            return 1.0
        
        # 中文平均语速约为每分钟150-200字，我们按照每秒2.5字计算（稍慢一些）
        chars_per_second = 2.5
        # 标点符号增加停顿时间
        punctuation_count = text.count('。') + text.count('！') + text.count('？') + text.count('，')
        punctuation_delay = punctuation_count * 0.4  # 每个标点增加0.4秒停顿
        
        # 基础播放时长 + 标点停顿 + 适度缓冲时间
        duration = len(text) / chars_per_second + punctuation_delay + 1.5
        
        # 🔥 重要修复：移除15秒硬限制，支持长文本！
        # 设置合理的最小值和最大值限制
        estimated = max(2.0, min(duration, 120.0))  # 最长支持2分钟
        
        # 详细的日志输出，帮助调试
        if len(text) > 50:  # 长文本时输出更详细的信息
            logger.info(f"📏 [长文本] 字数: {len(text)}, 标点: {punctuation_count}, "
                       f"基础时长: {len(text)/chars_per_second:.1f}s, "
                       f"标点延迟: {punctuation_delay:.1f}s, "
                       f"最终估算: {estimated:.1f}s")
        else:
            logger.info(f"📏 文本长度: {len(text)}字, 标点: {punctuation_count}个, 估算时长: {estimated:.1f}秒")
        
        return estimated

    def start_timeout_timer(self, duration, callback, message="操作超时。"):
        self.cancel_timeout_timer()
        self.conversation_timeout_timer = threading.Timer(duration, lambda: self.timeout_handler(callback, message))
        self.conversation_timeout_timer.start()
        logger.info(f"⏰ 启动定时器: {duration}秒 - {message}")

    def cancel_timeout_timer(self):
        if self.conversation_timeout_timer:
            self.conversation_timeout_timer.cancel()
            self.conversation_timeout_timer = None
            logger.info("🚫 已取消现有定时器")
            
    def timeout_handler(self, callback, message):
        logger.warning(f"⏰ 定时器触发: {message}")
        callback()

    def reset_to_idle(self):
        self.cancel_timeout_timer()
        if self.state != SystemState.IDLE:
            self.set_state(SystemState.IDLE)
            self.current_user = None
            
            # 延迟恢复视觉系统，防止立即重新触发
            def delayed_vision_resume():
                if self.state == SystemState.IDLE and self.vision_system:  # 确保仍在IDLE状态
                    self.vision_system.resume()
                    logger.info("🎛️ 已延迟恢复视觉模块")
                else:
                    logger.info("🎛️ 跳过视觉模块恢复 - 状态已改变")
            
            # 延迟2秒后恢复视觉模块
            resume_timer = threading.Timer(2.0, delayed_vision_resume)
            resume_timer.start()
            
            logger.info("系统已重置为空闲状态，将在2秒后恢复视觉模块。")

# ----------------- 其他模块适配 -----------------

from voice_llm.tts_module import TTSModule # 导入您的模块

class SimpleTTSWrapper:
    """
    集成了 TTSModule 的TTS模块包装器
    支持播放时长估算和延迟回调
    """
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.tts_module = None
        self.active_playbacks = {}  # 追踪活跃的播放任务
        self.active_timers = {}     # 追踪活跃的延迟定时器

    def _on_tts_event_callback(self, event_data):
        """
        当 TTSModule 完成播放或发生错误时，此函数被调用。
        
        🔧 关键修复：正确区分"音频生成完成"和"音频播放完成"
        """
        # 检查是否是音频生成完成的事件
        if event_data.get("source") == "voice" and "ok" in event_data:
            trace_id = event_data.get('trace_id')
            stopped = event_data.get('stopped', False)
            
            logger.info(f"🔊 TTS 音频生成完成: trace_id={trace_id}, stopped={stopped}")
            
            # ⚠️ 重要：这里只是音频生成完成，不是播放完成！
            # 需要等待实际的播放时长
            
            if trace_id in self.active_playbacks:
                playback_info = self.active_playbacks[trace_id]
                estimated_duration = playback_info['duration']
                text = playback_info['text']
                generation_time = time.time()
                start_time = playback_info['start_time']
                
                # 🎯 关键修复：从音频生成完成时刻开始计算播放延迟
                # 而不是从TTS命令发送时刻
                playback_info['generation_complete_time'] = generation_time
                
                # 计算到目前为止已经耗费的时间
                elapsed_since_start = generation_time - start_time
                
                # 🔧 保守估算：确保给足够时间让音频真正播放完毕
                # 从生成完成开始，再等待完整的播放时长
                remaining_playback_time = estimated_duration
                
                logger.info(f"🔊 音频生成耗时: {elapsed_since_start:.1f}s, "
                           f"预计还需播放: {remaining_playback_time:.1f}s "
                           f"(文本: '{text[:15]}...')")
                
                # 延迟发送播放完成回调
                def send_delayed_callback():
                    # 检查定时器是否被取消
                    if trace_id not in self.active_timers:
                        logger.info(f"🔊 ⏹️ 播放回调已被取消: {trace_id}")
                        return
                    
                    actual_total_time = time.time() - start_time
                    router_callback_msg = {
                        'status': 'finished',
                        'trace_id': trace_id,
                        'text': text,
                        'estimated_duration': estimated_duration,
                        'actual_total_time': actual_total_time,
                        'generation_time': elapsed_since_start,
                        'playback_completed': True
                    }
                    self.message_bus.send('tts_callback_to_router', router_callback_msg)
                    
                    # 清理记录
                    if trace_id in self.active_playbacks:
                        del self.active_playbacks[trace_id]
                    if trace_id in self.active_timers:
                        del self.active_timers[trace_id]
                    
                    logger.info(f"🔊 ✅ 播放真正完成: {trace_id}, 总耗时={actual_total_time:.1f}s "
                               f"(生成={elapsed_since_start:.1f}s + 播放={remaining_playback_time:.1f}s)")
                
                # 🎯 从音频生成完成时刻开始等待播放时长
                delay_timer = threading.Timer(remaining_playback_time, send_delayed_callback)
                self.active_timers[trace_id] = delay_timer  # 记录定时器
                delay_timer.start()
                
            else:
                # 没有预估时长记录的情况，立即发送回调
                logger.warning(f"⚠️ 未找到 {trace_id} 的播放记录，立即发送完成回调")
                router_callback_msg = {
                    'status': 'finished',
                    'trace_id': trace_id
                }
                self.message_bus.send('tts_callback_to_router', router_callback_msg)
        
        # 也可以在这里处理错误事件
        elif "error_code" in event_data:
             logger.error(f"❌ TTS 模块内部错误: {event_data['message']}")

    def estimate_tts_duration(self, text):
        """
        估算TTS播放时长 - 支持长文本
        """
        if not text:
            return 1.0
        
        # 中文平均语速约为每分钟150-200字，我们按照每秒2.5字计算（稍慢一些）
        chars_per_second = 2.4
        # 标点符号增加停顿时间
        punctuation_count = text.count('。') + text.count('！') + text.count('？') + text.count('，')
        punctuation_delay = punctuation_count * 0.4
        
        # 基础播放时长 + 标点停顿 + 系统播放延迟
        duration = len(text) / chars_per_second + punctuation_delay + 1.5
        
        estimated = max(2.0, min(duration, 120.0))  # 最长支持2分钟
        
        # 记录估算详情
        if len(text) > 50:
            logger.info(f"🔊 [TTS长文本] 字数: {len(text)}, 估算播放时长: {estimated:.1f}s")
        
        return estimated

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
                    text = msg.get('text', '')
                    trace_id = msg.get('trace_id', str(uuid.uuid4()))
                    
                    # 估算播放时长并记录
                    estimated_duration = self.estimate_tts_duration(text)
                    self.active_playbacks[trace_id] = {
                        'text': text,
                        'duration': estimated_duration,
                        'start_time': time.time()
                    }
                    
                    logger.info(f"🔊 开始TTS播报: '{text[:30]}...' (预估时长: {estimated_duration:.1f}秒)")
                    
                    # 将我们的消息格式转换为 TTSModule 期望的 schema 格式
                    tts_command = {
                        "ts": time.time(),
                        "trace_id": trace_id,
                        "text": text,
                        "priority": msg.get('priority', 5),
                        "interruptible": True
                    }
                    self.tts_module.handle_tts_say(tts_command)
                
                elif msg_type == 'tts_stop':
                    logger.info("🔊 🛑 收到TTS停止命令")
                    
                    # 🔥 立即取消所有延迟定时器，防止重复播报
                    cancelled_count = 0
                    for trace_id, timer in list(self.active_timers.items()):
                        timer.cancel()
                        del self.active_timers[trace_id]
                        cancelled_count += 1
                        logger.info(f"🔊 ⏹️ 已取消延迟回调: {trace_id}")
                    
                    if cancelled_count > 0:
                        logger.info(f"🔊 🛑 已取消 {cancelled_count} 个延迟播放回调")
                    
                    # 清理活跃播放记录
                    if self.active_playbacks:
                        logger.info(f"🔊 🧹 清理 {len(self.active_playbacks)} 个活跃播放记录")
                        self.active_playbacks.clear()
                    
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
    """
    音频处理模块 - 集成 FunASR (asr_service.py)
    
    这个模块会轮询路由器的状态，并根据状态控制ASR引擎的启停。
    """
    def __init__(self, message_bus, router, config=None):
        self.message_bus = message_bus
        self.router = router  # 需要路由器的引用来检查状态
        self.config = config  # 新增：配置参数
        
        # 获取ASR配置参数
        if config and hasattr(config, 'asr'):
            self.asr_config = config.asr
        else:
            # 默认配置 (向后兼容)
            self.asr_config = {
                'amplitude_threshold': 0.05,         # 原来的值，保持兼容
                'silence_threshold': 0.02,           # 原来的值
                'confidence_required': 3,            # 原来的值
                'confidence_decay': 1,
                'endpoint_silence_ms': 2000,         # 原来的值
                'debug_audio_threshold': 0.01,       # 原来的值
                'debug_silence_interval': 500,       # 原来的值
                'enable_pre_detection_recording': True,
                'pre_speech_buffer_ms': 500,
            }
        
        logger.info(f"🎛️ ASR参数配置: 幅度阈值={self.asr_config['amplitude_threshold']}, "
                   f"置信度要求={self.asr_config['confidence_required']}, "
                   f"静音检测={self.asr_config['endpoint_silence_ms']}ms")
        
        self.asr = None
        self.vad = None
        self.pa = None
        self.stream = None
        
        if not ASR_MODULE_LOADED:
            logger.error("ASR依赖未加载，AudioSystem 将无法工作。")
            return
            
        try:
            # 使用配置中的静音检测阈值
            ASR_CONFIG["endpoint_silence_ms"] = self.asr_config['endpoint_silence_ms']
            logger.info(f"🎤 ASR 静音检测阈值已调整为: {ASR_CONFIG['endpoint_silence_ms']}ms")
            
            self.pa = pyaudio.PyAudio()
            # 使用配置中的模型路径，如果有的话
            model_dir = None
            if self.config and hasattr(self.config, 'asr') and 'model_dir' in self.config.asr:
                model_dir = self.config.asr['model_dir']
                logger.info(f"🎤 使用本地ASR模型: {model_dir}")
            else:
                logger.info("🎤 将从远程下载ASR模型")
            
            self.asr = FunASRWrapper(model_dir=model_dir, device=ASR_CONFIG["device"])
            self.vad = SimpleVAD(ASR_CONFIG["sample_rate"])
        except Exception as e:
            logger.error(f"❌ 无法初始化 Pyaudio 或 FunASR: {e}", exc_info=True)
            if self.pa:
                self.pa.terminate()
            self.pa = None

    def start_processing(self):
        """
        音频处理入口函数
        
        只在LISTENING状态下工作，专注于ASR语音识别
        不再支持语音唤醒或语音打断检测
        
        支持的音频处理模式：
        - IDLE/GREETING/RESPONDING: 完全静默，不开启音频流
        - LISTENING: 全功能ASR语音识别
        """
        if not self.pa or not self.asr or not self.vad:
            logger.error("❌ AudioSystem 启动失败，核心组件未初始化。")
            return
            
        logger.info("🎤 音频模块启动 (仅LISTENING状态ASR模式)")
        
        # 全局音频流和状态变量
        sr = ASR_CONFIG["sample_rate"]
        chunk_samples = int(ASR_CONFIG["chunk_sec"] * sr)
        
        # ASR状态变量
        trailing_sil_ms = 0
        speech_detected = False
        speech_confidence = 0
        
        logger.info("🎤 说明: 系统只在按键唤醒后进入LISTENING状态时工作")
        
        while self.message_bus.is_running:
            try:
                current_state = self.router.state
                
                # === 只在LISTENING状态下工作 ===
                if current_state == SystemState.LISTENING:
                    
                    # 确保音频流已打开
                    if not self.stream:
                        try:
                            logger.info("🎤 进入LISTENING状态，启动ASR音频流...")
                            self.stream = self.pa.open(format=pyaudio.paInt16,
                                                      channels=1, rate=sr, input=True,
                                                      frames_per_buffer=chunk_samples)
                            
                            # 重置ASR状态变量
                            self.asr.recognize_final()  # 清空ASR缓冲区
                            trailing_sil_ms = 0
                            speech_detected = False
                            speech_confidence = 0
                            
                            logger.info("🎤 ASR音频流已启动，等待语音输入...")
                            
                        except Exception as e:
                            logger.error(f"❌ 打开麦克风失败: {e}", exc_info=True)
                            time.sleep(1)
                            continue
                    
                    # 读取音频数据并进行ASR处理
                    try:
                        pcm_bytes = self.stream.read(chunk_samples, exception_on_overflow=False)
                        data16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                        mono = (data16.astype(np.float32) / 32768.0).reshape(-1)
                        
                        # 音频信号强度检测
                        max_val = np.max(np.abs(mono))
                        vad_result = self.vad.is_speech(pcm_bytes)
                        
                        # 使用配置的检测阈值
                        amplitude_threshold = self.asr_config['amplitude_threshold']
                        amplitude_speech = max_val > amplitude_threshold
                        silence_threshold = self.asr_config['silence_threshold']
                        is_silence = max_val < silence_threshold

                        # 将音频追加到ASR
                        self.asr.append_audio(mono)
                        
                        # 基于连续性的语音检测
                        confidence_required = self.asr_config['confidence_required']
                        confidence_decay = self.asr_config['confidence_decay']
                        
                        if vad_result and amplitude_speech:
                            speech_confidence += 1
                        else:
                            speech_confidence = max(0, speech_confidence - confidence_decay)
                        
                        is_real_speech = speech_confidence >= confidence_required
                        
                        # 调试输出
                        debug_threshold = self.asr_config['debug_audio_threshold']
                        if max_val > debug_threshold:
                            logger.info(f"🎤 [ASR] 音频: {max_val:.4f}, VAD: {'√' if vad_result else '✗'}, "
                                       f"幅度: {'√' if amplitude_speech else '✗'}, 置信度: {speech_confidence}/{confidence_required}")
                        
                        # 语音端点检测
                        if is_real_speech and not is_silence:
                            if not speech_detected:
                                logger.info("🎤 ✅ 检测到连续语音开始！")
                                speech_detected = True
                            trailing_sil_ms = 0
                        else:
                            if speech_detected:
                                trailing_sil_ms += int(1000 * len(data16) / sr)
                                if trailing_sil_ms % self.asr_config['debug_silence_interval'] == 0:
                                    logger.info(f"🎤 静音累积: {trailing_sil_ms}ms")
                        
                        # 语音端点触发ASR识别
                        if speech_detected and trailing_sil_ms >= ASR_CONFIG["endpoint_silence_ms"]:
                            logger.info("🎤 语音端点检测 -> 执行ASR识别")
                            final_text = self.asr.recognize_final()
                            
                            logger.info(f"🎤 ASR结果: '{final_text}' (长度: {len(final_text) if final_text else 0})")
                            
                            self.message_bus.send('audio_to_router', {
                                'type': 'user_command',
                                'text': final_text if final_text else ""
                            })
                            
                            # 重置状态
                            trailing_sil_ms = 0
                            speech_detected = False
                        
                    except Exception as e:
                        logger.warning(f"⚠️ ASR音频处理出错: {e}")
                        self.stop_stream()
                        time.sleep(0.1)
                
                # === 其他状态：关闭音频流 ===
                else:  
                    if self.stream:
                        logger.info(f"🎤 退出LISTENING状态，关闭ASR音频流")
                        self.stop_stream()
                    time.sleep(0.2)  # 低功耗轮询

            except Exception as e:
                logger.error(f"❌ AudioSystem 主循环严重错误: {e}", exc_info=True)
                self.stop_stream()
                time.sleep(1)
        
        # 退出清理
        self.stop_stream()
        if self.pa:
            self.pa.terminate()

    def stop_stream(self):
        """辅助函数，用于安全关闭流"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.warning(f"关闭麦克风流时出错: {e}")
        self.stream = None

# vision processing module
from vision.face_reco import RealTimeFaceRecognition # 导入您的模块

class SimpleVisionSystem:
    """
    集成了 RealTimeFaceRecognition 的视觉模块包装器
    (已修复竞态条件 - 保留 Asyncio)
    """
    def __init__(self, message_bus, config):
        self.message_bus = message_bus
        self.config = config
        self.recognizer = None
        self.is_ready = False # 新增：用于解决竞态条件的标志
        self.loop = None      # 新增：保存循环的引用

    def _on_face_detected_callback(self, event_type, event_data):
        """
        当 RealTimeFaceRecognition 识别到人脸时，此函数被调用。
        它将事件数据发送到消息总线。
        """
        logger.info(f"👁️  视觉回调: 检测到 {event_data.get('keyword')}")
        self.message_bus.send('vision_to_router', event_data)

    def start_processing(self):
        """
        作为 Vision 线程的入口函数。
        """
        logger.info("👁️  正在初始化视觉模块...")
        try:
            # RealTimeFaceRecognition 内部管理着自己的处理线程和asyncio循环
            # 我们需要在一个新的事件循环中启动它
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            self.recognizer = RealTimeFaceRecognition(
                model_path=self.config.vision['model_path'],
                gallery_dir=self.config.vision['gallery_dir']
            )
            # 设置回调函数，将识别结果连接到我们的消息总线
            self.recognizer.set_direct_callback(self._on_face_detected_callback)
            
            logger.info("👁️  视觉模块初始化完成，启动处理循环...")
            
            # --- 关键修复 ---
            # 启动识别器（这将启动其内部的处理线程）
            # 这会调用 vision/face_reco.py 中的 async def start
            self.loop.run_until_complete(self.recognizer.start())
            # --- 修复结束 ---

            # --- 关键修复 ---
            # 通知其他线程本模块已准备就绪
            self.is_ready = True
            logger.info("👁️  视觉模块已标记为 Ready。")
            # --- 修复结束 ---
            
            # 让当前线程(Vision)保持存活，以接收暂停/恢复指令和运行asyncio任务
            # self.loop.run_forever() # 不使用 run_forever()，因为它会阻塞
            
            # 使用一个循环来保持线程存活，同时允许总线停止
            while self.message_bus.is_running:
                # 运行所有待处理的 asyncio 任务
                self.loop.call_soon(lambda: None) # 确保循环不是空的
                self.loop.run_until_complete(asyncio.sleep(1))

        except Exception as e:
            self.is_ready = True # 即使失败也要标记为 ready，以防 controller 死锁
            logger.error(f"❌ 视觉模块启动失败: {e}", exc_info=True)
        finally:
            if self.recognizer and self.recognizer.is_running:
                logger.info("👁️  停止视觉识别器...")
                if self.loop and self.message_bus.is_running:
                    # 这会调用 vision/face_reco.py 中的 async def stop
                    self.loop.run_until_complete(self.recognizer.stop())
            if self.loop:
                self.loop.close()
            logger.info("👁️  视觉线程已停止。")

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
    """系统控制器 - 简化版，仅用于向后兼容"""
    def __init__(self, message_bus, vision_system):
        self.message_bus = message_bus
        self.vision_system = vision_system
        
    def start_processing(self):
        logger.info("🎛️ 系统控制器启动...")
        # 由于现在使用直接控制，这个控制器主要用于兼容性
        # 实际的控制逻辑已经移到 StateMachineRouter 中
        
        while self.message_bus.is_running:
            try:
                msg = self.message_bus.receive('system_control', timeout=1.0)
                if msg is None:
                    continue

                # 保留这些方法以防有遗留的消息队列调用
                if msg.get('type') == 'pause_vision':
                    logger.info("🎛️ SystemController 收到 pause_vision（已弃用，建议使用直接调用）")
                    self.vision_system.pause()
                
                elif msg.get('type') == 'resume_vision':
                    logger.info("🎛️ SystemController 收到 resume_vision（已弃用，建议使用直接调用）")
                    self.vision_system.resume()
                    
            except Exception as e:
                logger.error(f"❌ SystemController 循环出错: {e}", exc_info=True)
                time.sleep(1)


# ----------------- 主程序 -----------------
class AppConfig:
    vision = {
        'model_path': r"C:/Users/17321/Desktop/Mirror/Smart-Entrance-Mirror-SELF/model/w600k_r50.onnx",
        'gallery_dir': r"C:/Users/17321/Desktop/Mirror/Smart-Entrance-Mirror-SELF/gallery_dataset"
    }
    
    # ASR配置 - 单次对话模式，极简配置
    asr = {
        'model_dir': r"C:/Users/17321/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        
        # 🎛️ 语音检测参数 (优化为快速响应)
        'amplitude_threshold': 0.025,       # 音频信号强度阈值 (稍微降低，提高敏感度)
        'silence_threshold': 0.012,         # 静音检测阈值   
        'confidence_required': 2,           # 连续检测次数要求 (快速确认)
        'confidence_decay': 1,              # 置信度衰减速度
        'endpoint_silence_ms': 1200,        # 语音结束静音时长 (减少延迟)
        
        # 🔍 调试参数
        'debug_audio_threshold': 0.008,     # 调试日志阈值
        'debug_silence_interval': 400,      # 静音调试间隔(ms)
        
        # 🚀 性能参数 
        'enable_pre_detection_recording': True, # 启用语音预录音
        'pre_speech_buffer_ms': 300,        # 语音前缓冲(ms) - 减少延迟
    }
    
    # ⌨️ 即时唤醒配置 (适配未来的语音/手势唤醒)
    keyboard = {
        'enable_keyboard_wakeup': True,     # 启用按键唤醒 (临时方案)
        'wakeup_keys': ['space', 'enter'],  # 唤醒按键 (空格/回车)
        # TODO: 未来扩展
        # 'enable_voice_wakeup': False,     # 语音唤醒
        # 'enable_gesture_wakeup': False,   # 手势唤醒  
    }
    
    # TTS配置
    tts = {}

class SmartMirrorApp:
    def __init__(self):
        self.message_bus = SimpleMessageBus()
        self.threads = []
        self.config = AppConfig()
        
        # 初始化所有模块
        self.vision = SimpleVisionSystem(self.message_bus, self.config)
        # 将 vision 引用传给 router，实现直接控制
        self.router = StateMachineRouter(self.message_bus, vision_system=self.vision, config=self.config)
        
        # 核心模块
        self.audio = AudioSystem(self.message_bus, self.router, self.config)  # ASR专用
        self.keyboard_listener = KeyboardWakeupListener(self.message_bus, self.config)  # 按键唤醒
        self.tts = SimpleTTSWrapper(self.message_bus)
        self.llm = SimpleLLMWrapper(self.message_bus)
        self.controller = SystemController(self.message_bus, self.vision)

        self.modules = {
            "Router": self.router,
            "Vision": self.vision,
            "Audio": self.audio,
            "KeyboardListener": self.keyboard_listener,  # 新增
            "TTS": self.tts,
            "LLM": self.llm,
            "Controller": self.controller
        }

    def start(self):
        logger.info("🚀 启动智能镜系统 (按键唤醒模式)...")
        
        # 按照依赖关系顺序启动模块
        startup_order = [
            ("Router", self.router),
            ("KeyboardListener", self.keyboard_listener),  # 按键监听优先启动
            ("TTS", self.tts),
            ("LLM", self.llm),
            ("Audio", self.audio),      # ASR模块
            ("Vision", self.vision),    # 视觉模块
            ("Controller", self.controller)
        ]
        
        for name, module in startup_order:
            logger.info(f"🔧 启动模块: {name}")
            thread = threading.Thread(target=module.start_processing, name=name, daemon=True)
            self.threads.append(thread)
            thread.start()
            time.sleep(0.1)  # 稍微延迟确保启动顺序
        
        logger.info("✅ 所有模块线程已启动")
        logger.info("🏠 智能家居交互模式 - 即时响应设计")
        logger.info("=" * 50)
        logger.info("📋 交互方式:")
        logger.info("   👁️ 人脸识别: 自动检测并问候已知用户")
        logger.info("   ⚡ 即时唤醒: 按空格/回车键立即激活语音")
        logger.info("   🗣️ 单次对话: 一次唤醒 -> 一次回复 -> 自动返回待机")
        logger.info("   🚨 随时打断: 任何时候按键都能立即打断并重新开始")
        logger.info("")
        logger.info("🎯 设计理念:")
        logger.info("   ⚡ 最短延迟: 无需等待，即按即应")
        logger.info("   🎭 自然交互: 模拟真实对话的打断机制") 
        logger.info("   � 简洁流程: 避免复杂的多轮对话状态")
        logger.info("   🏗️ 扩展就绪: 未来可切换到语音/手势唤醒")
        logger.info("=" * 50)
        logger.info("⏸️ 退出系统: 按 Ctrl+C")
        
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

def check_dependencies():
    """检查必要依赖并给出安装建议"""
    missing_deps = []
    
    if not ASR_MODULE_LOADED:
        missing_deps.append("ASR模块 (FunASR, pyaudio)")
    
    if not KEYBOARD_MODULE_LOADED:
        missing_deps.append("keyboard (pip install keyboard)")
    
    if missing_deps:
        logger.warning("⚠️ 检测到缺失依赖项:")
        for dep in missing_deps:
            logger.warning(f"   - {dep}")
        logger.warning("🔧 部分功能可能无法正常工作，建议安装对应依赖")
        return False
    return True

if __name__ == "__main__":
    logger.info("🔍 检查系统依赖...")
    check_dependencies()
    
    logger.info("🎯 启动智能镜系统（按键唤醒模式）...")
    app = SmartMirrorApp()
    app.start()