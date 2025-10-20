import json
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError
from constracts.events import SCHEMAS
import logging

logger = logging.getLogger(__name__)

class EventValidator:
    """事件数据验证器"""
    
    @staticmethod
    def validate_event(event_type: str, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证事件数据是否符合schema定义
        
        Args:
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            tuple[bool, Optional[str]]: (是否验证通过, 错误信息)
        """
        schema = SCHEMAS.get(event_type)
        if not schema:
            return False, f"Unknown event type: {event_type}"
            
        try:
            validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            error_msg = f"Validation failed for {event_type}: {e.message}"
            logger.error(error_msg)
            return False, error_msg
            
    @staticmethod
    def format_error_event(source: str, error_code: str, message: str, 
                          trace_id: str, retry_after_ms: Optional[int] = None) -> Dict[str, Any]:
        """格式化错误事件数据
        
        Args:
            source: 错误来源
            error_code: 错误代码
            message: 错误信息
            trace_id: 追踪ID
            retry_after_ms: 重试等待时间（毫秒）
            
        Returns:
            Dict[str, Any]: 格式化的错误事件数据
        """
        import time
        
        error_data = {
            "ts": time.time(),
            "trace_id": trace_id,
            "source": source,
            "error_code": error_code,
            "message": message
        }
        
        if retry_after_ms is not None:
            error_data["retry_after_ms"] = retry_after_ms
            
        return error_data