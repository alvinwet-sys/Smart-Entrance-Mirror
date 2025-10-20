#!/usr/bin/env python3
"""
LLM模块主程序
"""

import asyncio
import logging
import signal
import sys
from llm_interface import LLMInterface
from event_handler import EventHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_main")

class LLMModule:
    """LLM主模块"""
    
    def __init__(self):
        self.llm_interface = LLMInterface()
        self.event_handler = EventHandler(self.llm_interface)
        self.running = False
    
    async def start(self):
        """启动模块"""
        logger.info("启动LLM模块...")
        
        # 健康检查
        if not await self.llm_interface.health_check():
            logger.error("LLM健康检查失败")
            return False
        
        self.running = True
        logger.info("LLM模块启动成功")
        return True
    
    async def stop(self):
        """停止模块"""
        logger.info("停止LLM模块...")
        self.running = False
    
    async def process_message(self, message: str) -> str:
        """处理消息（模拟事件总线消息）"""
        try:
            event_data = json.loads(message)
            topic = event_data.get("topic")
            data = event_data.get("data", {})
            
            if topic == "core.decision_request":
                result = await self.event_handler.handle_event(topic, data)
                if result:
                    return json.dumps(result)
            
            return None
            
        except json.JSONDecodeError:
            logger.error("消息JSON解析失败")
            return None
        except Exception as e:
            logger.error(f"消息处理异常: {e}")
            return None

async def main():
    """主函数"""
    llm_module = LLMModule()
    
    if not await llm_module.start():
        sys.exit(1)
    
    # 信号处理
    def signal_handler(signum, frame):
        logger.info(f"接收到信号 {signum}，准备退出...")
        asyncio.create_task(llm_module.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 主循环（简化版）
    try:
        while llm_module.running:
            # 这里应该从事件总线接收消息
            # 简化实现：等待退出信号
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("接收到键盘中断")
    finally:
        await llm_module.stop()
        logger.info("LLM模块已停止")

if __name__ == "__main__":
    asyncio.run(main())