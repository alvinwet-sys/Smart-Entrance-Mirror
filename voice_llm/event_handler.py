import asyncio
import json
import logging
from typing import Callable, Dict, Any
from llm_interface import LLMInterface

logger = logging.getLogger("event_handler")

class EventHandler:
    """事件处理器"""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.event_handlers = {
            "core.decision_request": self._handle_decision_request
        }
    
    async def handle_event(self, topic: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理事件"""
        handler = self.event_handlers.get(topic)
        if not handler:
            logger.warning(f"未找到事件处理器: {topic}")
            return None
        
        try:
            return await handler(event_data)
        except Exception as e:
            logger.error(f"事件处理失败: {e}")
            return None
    
    async def _handle_decision_request(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理决策请求"""
        try:
            result = await self.llm_interface.process_decision_request(event_data)
            return {
                "topic": "llm.decision_ready",
                "data": result
            }
        except Exception as e:
            logger.error(f"决策请求处理失败: {e}")
            # 返回错误事件
            return {
                "topic": "core.error_event",
                "data": {
                    "ts": event_data['ts'],
                    "trace_id": event_data['trace_id'],
                    "source": "llm",
                    "error_code": "PROCESSING_ERROR",
                    "message": str(e),
                    "retry_after_ms": 5000
                }
            }