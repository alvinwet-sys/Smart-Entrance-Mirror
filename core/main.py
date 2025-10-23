import asyncio
import logging
from aiohttp import web

from config import load_cfg
from bus import EventBus
from router import Router
from healthcheck import build_app as build_hc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    try:
        # 加载配置
        cfg = load_cfg()
        logger.info("Configuration loaded")
        
        # 初始化事件总线
        bus = EventBus()
        logger.info("Event bus initialized")
        
        # 初始化路由器
        router = Router(cfg, bus.publish)
        
        # 初始化唤醒检测器
        from voice_llm import wakeup_detector
        wakeup = wakeup_detector.WakeupDetector(cfg, bus.publish)
        logger.info("Wakeup detector initialized")
        
        # 注册事件处理器
        bus.subscribe("face_id_resolved", router.on_face)
        bus.subscribe("voice.tts_done", router.on_tts_done)
        bus.subscribe("voice.asr_text", router.on_asr_text)
        bus.subscribe("voice.wakeup", router.on_wakeup)  
        bus.subscribe("llm.decision_ready", router.on_llm_decision)
        logger.info("Event handlers registered")
        
        # 启动健康检查服务
        app = await build_hc()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host="127.0.0.1", port=cfg.app.http_port)
        await site.start()
        logger.info(f"Health check service started on port {cfg.app.http_port}")
        
        # 启动唤醒检测器
        wakeup_task = asyncio.create_task(wakeup.start())
        logger.info("Wakeup detector started")
        
        # 启动事件总线
        bus_task = asyncio.create_task(bus.start())
        logger.info("Event bus started")
        
        # 保持主循环运行
        logger.info("[core] Smart Mirror System v1.0 started")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # 优雅关闭
            await wakeup.stop()
            logger.info("System shutdown completed")
            
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass