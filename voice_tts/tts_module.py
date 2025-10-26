# coding:UTF-8
from aip import AipSpeech
import os
import subprocess
import platform
import tempfile
import time
import json
import getpass
import threading
import queue
import sys
import uuid
import atexit

# 配置文件路径
CONFIG_FILE = os.path.expanduser("~/.baidu_tts_config.json")

# 错误代码定义
ERROR_CODES = {
    "CONFIG_LOAD_FAILED": "config_load_failed",
    "API_AUTH_FAILED": "api_auth_failed",
    "SYNTHESIS_FAILED": "synthesis_failed",
    "PLAYBACK_FAILED": "playback_failed",
    "VALIDATION_ERROR": "validation_error",
    "INTERNAL_ERROR": "internal_error",
    "TIMEOUT": "timeout",
    "RESOURCE_UNAVAILABLE": "resource_unavailable",
    "DEVICE_BUSY": "device_busy"
}

# 兼容旧版 jsonschema
try:
    from jsonschema import Draft7Validator
    HAS_DRAFT7 = True
except ImportError:
    # 回退到旧版验证
    from jsonschema import validate
    HAS_DRAFT7 = False

class TTSValidator:
    """TTS事件本地校验器（兼容旧版jsonschema）"""
    
    def __init__(self):
        self.schemas = self._load_schemas()
    
    def _load_schemas(self):
        """加载TTS事件Schema"""
        return {
            "core.tts_say": {
                "type": "object",
                "required": ["ts", "trace_id", "text", "priority"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "text": {"type": "string", "minLength": 1, "maxLength": 512},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 10},
                    "style": {"type": "string", "enum": ["default", "news", "cheerful", "calm"]},
                    "interruptible": {"type": "boolean"}
                },
                "additionalProperties": False
            },
            "core.tts_stop": {
                "type": "object",
                "required": ["ts", "trace_id", "source", "reason"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["core"]},
                    "reason": {"type": "string", "enum": ["preempt", "user", "error"]},
                    "timeout_ms": {"type": "integer", "minimum": 0},
                    "instance": {"type": "string"}
                },
                "additionalProperties": False
            },
            "voice.tts_done": {
                "type": "object",
                "required": ["ts", "trace_id", "source", "ok", "stopped"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["voice"]},
                    "ok": {"type": "boolean"},
                    "stopped": {"type": "boolean"}
                },
                "additionalProperties": False
            },
            "core.error_event": {
                "type": "object",
                "required": ["ts", "trace_id", "source", "error_code", "message"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "source": {"type": "string"},
                    "error_code": {"type": "string"},
                    "message": {"type": "string"},
                    "retry_after_ms": {"type": "integer", "minimum": 0}
                },
                "additionalProperties": False
            }
        }
    
    def validate_event(self, event_data, event_type):
        """验证事件数据"""
        if event_type not in self.schemas:
            return False, [f"未知的事件类型: {event_type}"]
        
        schema = self.schemas[event_type]
        
        if HAS_DRAFT7:
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(event_data))
            if errors:
                sorted_errors = sorted(errors, key=lambda e: e.path)
                error_messages = []
                for error in sorted_errors:
                    error_path = " -> ".join(str(p) for p in error.path) if error.path else "root"
                    error_messages.append(f"{error_path}: {error.message}")
                return False, error_messages
            return True, []
        else:
            # 旧版jsonschema兼容
            try:
                validate(event_data, schema)
                return True, []
            except Exception as e:
                return False, [str(e)]
    
    def validate_and_raise(self, event_data, event_type):
        """验证事件数据，失败则抛出异常"""
        is_valid, errors = self.validate_event(event_data, event_type)
        if not is_valid:
            error_msg = f"Schema校验失败 ({event_type}): " + "; ".join(errors)
            raise ValueError(error_msg)

class DeviceLock:
    """跨平台设备锁实现（使用文件锁）"""
    
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_handle = None
        self.is_locked = False
       
    
    def acquire(self, timeout=5.0):
        """获取设备锁"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # 尝试创建锁文件
                if os.path.exists(self.lock_file):
                    # 锁文件已存在，等待
                    time.sleep(0.1)
                    continue
                
                # 创建锁文件
                with open(self.lock_file, 'w') as f:
                    f.write(str(os.getpid()))
                self.is_locked = True
                return True
            except Exception:
                time.sleep(0.1)
        
        return False
    
    def release(self):
        """释放设备锁"""
        if self.is_locked:
            try:
                if os.path.exists(self.lock_file):
                    os.unlink(self.lock_file)
                self.is_locked = False
                return True
            except Exception:
                return False
        return False

class TTSModule:
    """TTS模块 - 可被集成到voice_and_llm_worker进程中"""
    
    def __init__(self, event_callback=None, config_file=None):
        """
        初始化TTS模块
        
        Args:
            event_callback: 事件回调函数，用于向主进程发送事件
            config_file: 配置文件路径，默认为 ~/.baidu_tts_config.json
        """
        self.event_callback = event_callback or self._default_event_callback
        self.config_file = config_file or CONFIG_FILE
        
        # 初始化校验器
        self.validator = TTSValidator()
        
        # 初始化设备锁
        self.device_lock = DeviceLock(os.path.join(tempfile.gettempdir(), "voice_device.lock"))
        
        # 加载配置
        config = self._load_config()
        if not config:
            raise Exception("TTS配置加载失败")
        
        # 初始化百度语音客户端
        try:
            self.client = AipSpeech(config['APP_ID'], config['API_KEY'], config['SECRET_KEY'])
        except Exception as e:
            self._send_error_event(
                trace_id="init",
                error_code=ERROR_CODES["API_AUTH_FAILED"],
                message=f"API初始化失败: {str(e)}"
            )
            raise
        
        # 消息队列和状态管理
        self.audio_queue = queue.PriorityQueue()
        self.current_task = None
        self.is_playing = False
        self.should_stop = False
        self.playback_process = None
        
        # 启动播放工作线程
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        # 注册退出清理
        atexit.register(self.cleanup)
    
    def _default_event_callback(self, event_data):
        """默认事件回调（打印到标准输出）"""
        print(f"[TTS_EVENT] {json.dumps(event_data, ensure_ascii=False)}")
        sys.stdout.flush()
    
    def _send_event(self, event_data):
        """发送事件到主进程"""
        if self.event_callback:
            self.event_callback(event_data)
    
    def _send_error_event(self, trace_id, error_code, message, retry_after_ms=None):
        """发送错误事件"""
        error_event = {
            "ts": time.time(),
            "trace_id": trace_id,
            "source": "voice",
            "error_code": error_code,
            "message": message
        }
        
        if retry_after_ms is not None:
            error_event["retry_after_ms"] = retry_after_ms
        
        # 本地校验错误事件
        try:
            self.validator.validate_and_raise(error_event, "core.error_event")
        except ValueError as e:
            return
        
        self._send_event(error_event)
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self._send_error_event(
                trace_id="init",
                error_code=ERROR_CODES["CONFIG_LOAD_FAILED"],
                message=f"配置文件加载失败: {str(e)}"
            )
            return None
        
        # 如果配置文件不存在，提示用户输入
        print("首次使用，请输入百度语音服务的API凭证")
        try:
            app_id = input("APP_ID: ").strip()
            api_key = input("API_KEY: ").strip()
            secret_key = getpass.getpass("SECRET_KEY: ").strip()
            
            if not all([app_id, api_key, secret_key]):
                self._send_error_event(
                    trace_id="init",
                    error_code=ERROR_CODES["CONFIG_LOAD_FAILED"],
                    message="API凭证不能为空"
                )
                return None
            
            # 保存配置
            config = {
                'APP_ID': app_id,
                'API_KEY': api_key,
                'SECRET_KEY': secret_key
            }
            
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(config, f)
                # 设置文件权限（Unix-like系统）
                if platform.system() != "Windows":
                    os.chmod(self.config_file, 0o600)
                return config
            except Exception:
                return config
                
        except Exception as e:
            self._send_error_event(
                trace_id="init",
                error_code=ERROR_CODES["CONFIG_LOAD_FAILED"],
                message=f"配置输入异常: {str(e)}"
            )
            return None
    
    def handle_tts_say(self, command):
        """处理core.tts_say命令"""
        # 本地校验命令格式
        try:
            self.validator.validate_and_raise(command, "core.tts_say")
        except ValueError as e:
            self._send_error_event(
                trace_id=command.get("trace_id", "unknown"),
                error_code=ERROR_CODES["VALIDATION_ERROR"],
                message=f"命令格式错误: {str(e)}"
            )
            return False
        
        # 设置默认值
        command.setdefault("style", "default")
        command.setdefault("interruptible", True)
        
        # 检查文本长度
        if len(command["text"].encode('utf-8')) > 1024:
            self._send_error_event(
                trace_id=command["trace_id"],
                error_code=ERROR_CODES["VALIDATION_ERROR"],
                message="文本长度超过1024字节限制"
            )
            return False
        
        # 添加到优先级队列
        priority = 10 - command["priority"]  # 转换为最小堆优先级
        self.audio_queue.put((priority, command))
        
        return True
    
    def handle_tts_stop(self, command):
        """处理core.tts_stop命令"""
        # 本地校验命令格式
        try:
            self.validator.validate_and_raise(command, "core.tts_stop")
        except ValueError as e:
            self._send_error_event(
                trace_id=command.get("trace_id", "unknown"),
                error_code=ERROR_CODES["VALIDATION_ERROR"],
                message=f"停止命令格式错误: {str(e)}"
            )
            return False
        
        self.should_stop = True
        
        # 停止当前播放进程
        if self.playback_process and self.playback_process.poll() is None:
            try:
                timeout_ms = command.get("timeout_ms", 300)
                self.playback_process.terminate()
                self.playback_process.wait(timeout=timeout_ms / 1000)
            except subprocess.TimeoutExpired:
                try:
                    self.playback_process.kill()
                    self.playback_process.wait()
                except Exception as e:
                    self._send_error_event(
                        trace_id=command["trace_id"],
                        error_code=ERROR_CODES["RESOURCE_UNAVAILABLE"],
                        message=f"强制停止播放进程失败: {str(e)}"
                    )
        
        # 清空队列（只保留非中断任务）
        temp_queue = queue.PriorityQueue()
        while not self.audio_queue.empty():
            priority, task = self.audio_queue.get()
            if not task.get("interruptible", True):
                temp_queue.put((priority, task))
        
        self.audio_queue = temp_queue
        self.is_playing = False
        self.current_task = None
        
        # 发送停止完成事件
        if command.get("trace_id"):
            self._send_tts_done(command["trace_id"], ok=False, stopped=True)
        
        return True
    
    def _send_tts_done(self, trace_id, ok=True, stopped=False):
        """发送voice.tts_done事件"""
        event = {
            "ts": time.time(),
            "trace_id": trace_id,
            "source": "voice",
            "ok": ok,
            "stopped": stopped
        }
        
        # 本地校验事件格式
        try:
            self.validator.validate_and_raise(event, "voice.tts_done")
        except ValueError as e:
            self._send_error_event(
                trace_id=trace_id,
                error_code=ERROR_CODES["INTERNAL_ERROR"],
                message=f"事件生成错误: {str(e)}"
            )
            return
        
        self._send_event(event)
    
    def _playback_worker(self):
        """播放工作线程"""
        while True:
            try:
                if self.audio_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # 获取最高优先级任务
                priority, task = self.audio_queue.get()
                self.current_task = task
                self.is_playing = True
                self.should_stop = False
                
                # 获取设备锁
                if not self.device_lock.acquire():
                    self._send_error_event(
                        trace_id=task["trace_id"],
                        error_code=ERROR_CODES["DEVICE_BUSY"],
                        message="音频设备忙，获取设备锁超时"
                    )
                    self._send_tts_done(task["trace_id"], ok=False, stopped=False)
                    self.audio_queue.task_done()
                    continue
                
                try:
                    # 合成语音
                    success, file_path = self._synthesize_speech(task)
                    if not success:
                        self._send_tts_done(task["trace_id"], ok=False, stopped=False)
                        self.audio_queue.task_done()
                        continue
                    
                    # 播放音频
                    play_success = self._play_audio(file_path, task)
                    
                    # 发送完成事件
                    self._send_tts_done(
                        task["trace_id"], 
                        ok=play_success and not self.should_stop, 
                        stopped=self.should_stop
                    )
                    
                finally:
                    # 释放设备锁
                    self.device_lock.release()
                
                self.is_playing = False
                self.current_task = None
                self.audio_queue.task_done()
                
            except Exception as e:
                error_msg = f"播放工作线程异常: {str(e)}"
                self._send_error_event(
                    trace_id=self.current_task["trace_id"] if self.current_task else "unknown",
                    error_code=ERROR_CODES["INTERNAL_ERROR"],
                    message=error_msg
                )
                
                if self.current_task:
                    self._send_tts_done(self.current_task["trace_id"], ok=False, stopped=False)
                
                self.is_playing = False
                self.current_task = None
                self.device_lock.release()  # 确保释放锁
    
    def _synthesize_speech(self, task):
        """合成语音"""
        try:
            # 语音参数映射
            voice_params = {
                "default": {"per": 0, "vol": 5},
                "news": {"per": 1, "vol": 5},
                "cheerful": {"per": 103, "vol": 7},
                "calm": {"per": 4, "vol": 3}
            }
            
            params = voice_params.get(task["style"], voice_params["default"])
            
            result = self.client.synthesis(
                task["text"],
                'zh', 
                1, 
                params
            )
            
            if isinstance(result, dict):
                error_msg = f"语音合成失败: {result.get('err_msg', '未知错误')}, 错误码: {result.get('err_no', '未知')}"
                self._send_error_event(
                    trace_id=task["trace_id"],
                    error_code=ERROR_CODES["SYNTHESIS_FAILED"],
                    message=error_msg,
                    retry_after_ms=5000  # 5秒后重试
                )
                return False, None
            
            # 创建临时文件（不自动删除）
            file_id = str(uuid.uuid4())
            temp_file_path = os.path.join(tempfile.gettempdir(), f"tts_{file_id}.mp3")
            with open(temp_file_path, 'wb') as f:
                f.write(result)
            
            return True, temp_file_path
                
        except Exception as e:
            error_msg = f"语音合成异常: {str(e)}"
            self._send_error_event(
                trace_id=task["trace_id"],
                error_code=ERROR_CODES["SYNTHESIS_FAILED"],
                message=error_msg
            )
            return False, None
    
    def _play_audio(self, file_path, task):
        """播放音频文件"""
        try:
            system = platform.system()
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self._send_error_event(
                    trace_id=task["trace_id"],
                    error_code=ERROR_CODES["PLAYBACK_FAILED"],
                    message=f"音频文件不存在: {file_path}"
                )
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self._send_error_event(
                    trace_id=task["trace_id"],
                    error_code=ERROR_CODES["PLAYBACK_FAILED"],
                    message="音频文件为空"
                )
                return False
            
            # Windows 平台
            if system == 'Windows':
                try:
                    # 使用Windows Media Player命令行工具直接播放
                    try:
                        self.playback_process = subprocess.Popen(
                            ['wmplayer', file_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    except FileNotFoundError:
                        # 如果wmplayer不可用，使用start命令
                        self.playback_process = subprocess.Popen(
                            ['start', file_path],
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    
                    # 等待播放完成或被打断
                    start_time = time.time()
                    timeout = 30  # 30秒超时
                    
                    while self.playback_process.poll() is None:
                        if self.should_stop:
                            self.playback_process.terminate()
                            return False
                        
                        if time.time() - start_time > timeout:
                            self._send_error_event(
                                trace_id=task["trace_id"],
                                error_code=ERROR_CODES["TIMEOUT"],
                                message="播放超时"
                            )
                            self.playback_process.terminate()
                            return False
                        
                        time.sleep(0.1)
                    
                    return True
                except Exception:
                    pass
            
            # macOS平台
            elif system == 'Darwin':
                self.playback_process = subprocess.Popen(
                    ['afplay', file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 等待播放完成或被打断
                start_time = time.time()
                timeout = 30  # 30秒超时
                
                while self.playback_process.poll() is None:
                    if self.should_stop:
                        self.playback_process.terminate()
                        return False
                    
                    if time.time() - start_time > timeout:
                        self._send_error_event(
                            trace_id=task["trace_id"],
                            error_code=ERROR_CODES["TIMEOUT"],
                            message="播放超时"
                        )
                        self.playback_process.terminate()
                        return False
                    
                    time.sleep(0.1)
                
                return True
            
            # Linux平台
            else:
                # 尝试多种播放方式
                try:
                    self.playback_process = subprocess.Popen(
                        ['aplay', file_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                except FileNotFoundError:
                    try:
                        self.playback_process = subprocess.Popen(
                            ['mpg123', file_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    except FileNotFoundError:
                        self.playback_process = subprocess.Popen(
                            ['ffplay', '-autoexit', '-nodisp', file_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                
                # 等待播放完成或被打断
                start_time = time.time()
                timeout = 30  # 30秒超时
                
                while self.playback_process.poll() is None:
                    if self.should_stop:
                        self.playback_process.terminate()
                        return False
                    
                    if time.time() - start_time > timeout:
                        self._send_error_event(
                            trace_id=task["trace_id"],
                            error_code=ERROR_CODES["TIMEOUT"],
                            message="播放超时"
                        )
                        self.playback_process.terminate()
                        return False
                    
                    time.sleep(0.1)
                
                return True
            
        except Exception as e:
            error_msg = f"播放失败: {str(e)}"
            self._send_error_event(
                trace_id=task["trace_id"],
                error_code=ERROR_CODES["PLAYBACK_FAILED"],
                message=error_msg
            )
            
            # 尝试使用 playsound 作为后备方案
            try:
                import playsound
                playsound.playsound(file_path, block=True)
                return True
            except ImportError:
                error_msg = "请安装 playsound 库: pip install playsound"
                self._send_error_event(
                    trace_id=task["trace_id"],
                    error_code=ERROR_CODES["RESOURCE_UNAVAILABLE"],
                    message=error_msg
                )
            except Exception:
                error_msg = "playsound 播放失败"
                self._send_error_event(
                    trace_id=task["trace_id"],
                    error_code=ERROR_CODES["PLAYBACK_FAILED"],
                    message=error_msg
                )
            
            return False
    
    def is_busy(self):
        """检查TTS是否正在工作"""
        return self.is_playing or not self.audio_queue.empty()
    
    def wait_until_done(self):
        """等待所有任务完成"""
        self.audio_queue.join()
    
    def cleanup(self):
        """清理资源"""
        # 停止当前播放
        if self.playback_process and self.playback_process.poll() is None:
            self.playback_process.terminate()
        
        # 释放设备锁
        self.device_lock.release()

# 使用示例
if __name__ == '__main__':
    # 简单的独立测试
    def event_handler(event_data):
        print(f"[主进程收到事件] {json.dumps(event_data, ensure_ascii=False)}")
    
    try:
        tts = TTSModule(event_callback=event_handler)
        
        # 测试命令
        test_say = {
            "ts": time.time(),
            "trace_id": "test_001",
            "text": "这是一个测试消息",
            "priority": 5,
            "style": "default",
            "interruptible": True
        }
        
        tts.handle_tts_say(test_say)
        
        # 等待播放完成
        time.sleep(10)
        
    except Exception:
        pass