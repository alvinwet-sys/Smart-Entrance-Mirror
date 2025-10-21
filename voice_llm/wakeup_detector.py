import asyncio
import logging
import queue
import sounddevice as sd
import numpy as np
from typing import Callable, Awaitable, Dict, Any
from datetime import datetime

from .wakeup_core import DTWKWS, VADHelper, CONFIG, int16_from_float32

logger = logging.getLogger(__name__)

class WakeupDetector:
    """语音唤醒检测器"""
    
    def __init__(self, config: Dict[str, Any], publish: Callable[[str, dict], Awaitable[bool]]):
        """初始化唤醒检测器
        
        Args:
            config: 配置信息
            publish: 事件发布函数
        """
        self.config = config
        self.publish = publish
        self.running = False
        self._audio_queue = queue.Queue()
        
        # 初始化组件
        self._setup_components()
        
    def _setup_components(self):
        """初始化检测组件"""
        try:
            # 初始化关键词检测
            self.kws = DTWKWS(
                keyword_paths=CONFIG["keywords"],
                dtw_threshold=CONFIG["dtw_threshold"],
                sample_rate=CONFIG["sample_rate"],
                frame_length=CONFIG["frame_length"],
                max_window_sec=CONFIG["max_window_sec"],
                min_window_sec=CONFIG["min_window_sec"],
                cooldown_sec=CONFIG["cooldown_sec"],
            )
            
            # 初始化VAD
            self.vad = VADHelper(
                sample_rate=CONFIG["sample_rate"],
                aggressiveness=CONFIG["vad_aggressiveness"],
                majority_frames=CONFIG["vad_majority_frames"],
                frame_ms=CONFIG["vad_frame_ms"]
            )
            
            # 准备关键词名称列表
            keyword_names_cfg = CONFIG.get("keyword_names", [])
            if keyword_names_cfg and len(keyword_names_cfg) == len(CONFIG["keywords"]):
                self.keyword_names = keyword_names_cfg
            else:
                self.keyword_names = [os.path.splitext(os.path.basename(p))[0] 
                                    for p in CONFIG["keywords"]]
                
        except Exception as e:
            logger.error(f"初始化检测组件失败: {e}")
            raise
            
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            logger.warning(f"音频回调状态: {status}")
        try:
            self._audio_queue.put(indata.copy(), block=False)
        except Exception as e:
            logger.debug(f"音频队列满或错误: {e}")
            
    async def _emit_error(self, error_code: str, message: str, retry_after_ms: int = None):
        """发送错误事件"""
        event = {
            "ts": datetime.now().timestamp(),
            "trace_id": "wakeup-" + str(int(datetime.now().timestamp() * 1000)),
            "source": "voice.wakeup",
            "error_code": error_code,
            "message": str(message)
        }
        if retry_after_ms is not None:
            event["retry_after_ms"] = retry_after_ms
            
        await self.publish("core.error_event", event)
        
    async def _process_frame(self, pcm16: np.ndarray, vad_flag: bool):
        """处理一帧音频数据"""
        frame_length = CONFIG["frame_length"]
        idx = 0
        while idx + frame_length <= len(pcm16):
            frame = pcm16[idx:idx + frame_length]
            idx += frame_length
            
            # 进行关键词检测
            res = self.kws.process(frame)
            if res >= 0:
                # 检测到唤醒词
                confidence = float(self.kws.get_last_confidence())
                event = {
                    "ts": datetime.now().timestamp(),
                    "trace_id": "wakeup-" + str(int(datetime.now().timestamp() * 1000)),
                    "source": "voice",
                    "keyword": self.keyword_names[res],
                    "confidence": confidence,
                    "channel": CONFIG["channel"],
                    "vad": vad_flag
                }
                
                # 发送唤醒事件
                await self.publish("voice.wakeup", event)
                logger.info(f"检测到唤醒词：{event['keyword']} (conf={confidence:.3f})")
                
    async def start(self):
        """启动唤醒检测"""
        if self.running:
            return
            
        self.running = True
        
        try:
            # 创建音频输入流
            stream = sd.InputStream(
                samplerate=CONFIG["sample_rate"],
                channels=1,
                dtype="float32",
                blocksize=CONFIG["frame_length"],
                callback=self._audio_callback,
                device=CONFIG["device"]
            )
            
            logger.info("唤醒检测器已启动，等待唤醒词...")
            
            with stream:
                pcm_buf = bytearray()
                
                while self.running:
                    try:
                        # 获取音频数据
                        indata = self._audio_queue.get(timeout=5.0)
                        pcm16 = int16_from_float32(indata[:, 0])
                        
                        # VAD 检测
                        pcm_buf.extend(pcm16.tobytes())
                        vad_flag = self.vad.is_speech(bytes(pcm_buf))
                        
                        # 处理音频帧
                        await self._process_frame(pcm16, vad_flag)
                        
                        # 控制 VAD 缓冲区大小
                        if len(pcm_buf) > CONFIG["sample_rate"] * 2:
                            pcm_buf = pcm_buf[-CONFIG["sample_rate"] * 2:]
                            
                    except queue.Empty:
                        await self._emit_error(
                            "AUDIO_STREAM_TIMEOUT",
                            "超过5秒未收到音频数据",
                            retry_after_ms=2000
                        )
                    except Exception as e:
                        await self._emit_error(
                            "PROCESSING_ERROR",
                            f"音频处理异常: {e}"
                        )
                        
        except Exception as e:
            await self._emit_error(
                "AUDIO_STREAM_ERROR",
                f"音频流创建失败: {e}"
            )
            self.running = False
            raise
            
    async def stop(self):
        """停止唤醒检测"""
        self.running = False
        logger.info("唤醒检测器已停止")