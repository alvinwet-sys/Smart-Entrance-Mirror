from dataclasses import dataclass
from typing import Optional

@dataclass
class Thresholds:
    face_confidence: float = 0.8
    asr_confidence: float = 0.7
    wakeup_confidence: float = 0.6

@dataclass
class Limits:
    vision_max: int = 100  # 视觉模块队列最大长度
    llm_max: int = 50     # LLM模块队列最大长度
    
@dataclass
class AppConfig:
    http_port: int = 8080
    
@dataclass
class Config:
    app: AppConfig
    limits: Limits
    thresholds: Thresholds
    
def load_cfg() -> Config:
    """加载配置"""
    return Config(
        app=AppConfig(http_port=8080),
        limits=Limits(vision_max=100, llm_max=50),
        thresholds=Thresholds(
            face_confidence=0.8,
            asr_confidence=0.7,
            wakeup_confidence=0.6
        )
    )