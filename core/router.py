from datetime import datetime
from typing import Callable, Awaitable, Any, Optional
import logging
from config import Config

logger = logging.getLogger(__name__)

class Router:
    def __init__(self, cfg: Config, publish: Callable[[str, dict], Awaitable[bool]]):
        self.cfg = cfg
        self.publish = publish
        self._current_face: Optional[str] = None
        self._last_asr: Optional[str] = None
        
    async def on_face(self, evt: dict):
        """处理人脸识别结果"""
        identity = evt.get('keyword')
        confidence = evt.get('confidence', 0)
        
        if confidence > self.cfg.thresholds.face_confidence:
            if identity != self._current_face:
                self._current_face = identity
                welcome_text = f"你好，{identity}!"
                
                await self.publish("core.tts_say", {
                    "ts": datetime.now().timestamp(),
                    "trace_id": evt['trace_id'],
                    "text": welcome_text,
                    "priority": "high"
                })
    
    async def on_tts_done(self, evt: dict):
        """处理TTS播放完成事件"""
        if not evt.get('ok', False):
            logger.error(f"TTS failed: {evt.get('error_message', 'unknown error')}")
            return
            
        # 如果是被主动停止的，不需要处理
        if evt.get('stopped', False):
            return
            
        # 这里可以处理一些TTS播放完成后的逻辑
        pass
        
    async def on_wakeup(self, evt: dict):
        """处理唤醒事件"""
        keyword = evt.get('keyword', '')
        confidence = evt.get('confidence', 0)
        
        if confidence >= self.cfg.thresholds.wakeup_confidence:
            # 发送欢迎提示音
            await self.publish("core.tts_say", {
                "ts": evt['ts'],
                "trace_id": evt['trace_id'],
                "text": "我在听",
                "priority": "high"
            })
            logger.info(f"唤醒成功：{keyword} (conf={confidence:.3f})")
        else:
            logger.debug(f"唤醒置信度不足：{keyword} (conf={confidence:.3f})")
        
    async def on_asr_text(self, evt: dict):
        """处理语音识别文本"""
        text = evt.get('text', '').strip()
        confidence = evt.get('confidence', 0)
        
        if not text or confidence < self.cfg.thresholds.asr_confidence:
            return
            
        self._last_asr = text
        
        await self.publish("core.decision_request", {
            "ts": evt['ts'],
            "trace_id": evt['trace_id'],
            "query": text,
            "context": {
                "identity": self._current_face,
                "last_asr": self._last_asr
            }
        })
        
    async def on_llm_decision(self, evt: dict):
        """处理LLM决策结果"""
        reply_text = evt.get('reply_text')
        if not reply_text:
            return
            
        await self.publish("core.tts_say", {
            "ts": evt['ts'],
            "trace_id": evt['trace_id'],
            "text": reply_text,
            "priority": evt.get('priority', 7)
        })
        
        # 处理额外动作
        for action in evt.get('actions', []):
            # TODO: 实现动作处理逻辑
            logger.info(f"Executing action: {action}")