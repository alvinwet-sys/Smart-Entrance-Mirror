import asyncio
from collections import defaultdict
from typing import Callable, Dict, List, Any, Coroutine, Union
import json
from constracts.events import SCHEMAS

class Event:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data
        self.priority = data.get("priority", "normal")
        
    @staticmethod
    def validate(type: str, data: dict) -> bool:
        schema = SCHEMAS.get(type)
        if not schema:
            return False
        try:
            # 这里应该实现JSON Schema验证
            return True
        except:
            return False

class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Event], Union[None, Coroutine[Any, Any, None]]]]] = defaultdict(list)
        self._q: asyncio.PriorityQueue[tuple[int, int, Event]] = asyncio.PriorityQueue()
        self._seq = 0


    @staticmethod
    def _prio_to_num(p: str) -> int:
        return {"critical": 0, "high": 1, "normal": 5, "low": 9}.get(p, 5)

    async def publish(self, type: str, data: dict) -> bool:
        """发布一个事件
        Args:
            type: 事件类型
            data: 事件数据
        Returns:
            bool: 是否成功发布
        """
        from core.validator import EventValidator
        
        # 验证事件数据
        is_valid, error_msg = EventValidator.validate_event(type, data)
        if not is_valid:
            # 如果验证失败，发布错误事件
            error_data = EventValidator.format_error_event(
                source="bus",
                error_code="INVALID_EVENT_DATA",
                message=error_msg,
                trace_id=data.get("trace_id", "unknown")
            )
            
            error_evt = Event("core.error_event", error_data)
            await self._q.put((self._prio_to_num("critical"), self._seq, error_evt))
            return False
            
        evt = Event(type, data)
        self._seq += 1
        await self._q.put((self._prio_to_num(evt.priority), self._seq, evt))
        return True

    def subscribe(self, event_type: str, handler: Callable[[Event], Union[None, Coroutine[Any, Any, None]]]):
        """订阅一个事件
        Args:
            event_type: 事件类型，支持通配符如 "core.*"
            handler: 事件处理函数，可以是同步或异步函数
        """
        self._subs[event_type].append(handler)
        
    def unsubscribe(self, event_type: str, handler: Callable[[Event], Union[None, Coroutine[Any, Any, None]]]):
        """取消订阅一个事件
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        if event_type in self._subs:
            self._subs[event_type].remove(handler)

    async def start(self):
        """启动事件总线的事件循环"""
        while True:
            try:
                _, _, evt = await self._q.get()
                # 广播给精确匹配的订阅者；支持通配符简化："core.*"
                base_type = evt.type.split('.')[0]
                handlers = (
                    self._subs.get(evt.type, []) +  # 精确匹配
                    self._subs.get(f"{base_type}.*", []) +  # 通配符匹配
                    self._subs.get("*", [])  # 全局匹配
                )
                
                for h in handlers:
                    try:
                        res = h(evt)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        # 发布错误事件
                        await self.publish("core.error_event", {
                            "ts": evt.data.get("ts", 0),
                            "trace_id": evt.data.get("trace_id", ""),
                            "source": evt.type,
                            "error_code": "HANDLER_ERROR",
                            "message": str(e)
                        })
            except Exception as e:
                print(f"[bus] event loop error: {e}")
                continue