import os
import yaml
from typing import Dict, Any

class Config:
    """配置管理类"""
    
    def __init__(self):
        # 默认配置
        self.defaults = {
            'timeouts': {
                'llm_ms': 8000,
                'mcp_ms': 5000,
                'api_timeout': 10
            },
            'priorities': {
                'llm': 7
            },
            'apis': {
                'deepseek': {
                    'api_key': 'sk-bdd73529195f4034961136f95e61e13d',
                    'base_url': 'https://api.deepseek.com/v1/chat/completions'
                },
                'juhe_mcp': {
                    'command': 'uvx',
                    'args': [
                        'juhe-mcp-proxy',
                        'https://mcp.juhe.cn/sse?token=r6fdf23P0ZkhlX1RDulv9AIwqPkoVPRtkUOugd1qqwhV5Y'
                    ],
                    'env': {
                        'LOG_LEVEL': 'CRITICAL'
                    }
                },
                'weather': {
                    'api_key': 'd77e2863faae221ab99eec3a42b72474'
                },
                'air_quality': {
                    'api_key': 'f539234597c17a6f098be7f8bb866d09'
                },
                'news': {
                    'api_key': '0a30380cc45df2eab487d1c6305df60b'
                }
            },
            'prompts': {
                'system_prompt': """你是一个智能家居助手，具有以下能力：
1. 日常对话交流
2. 查询和回复当前日期时间
3. 查询天气预报（需要用户提供城市）
4. 查询全国空气质量状况
5. 查询今日头条新闻

请根据用户需求提供准确、有用的回复。如果用户询问天气、空气质量或新闻，请明确说明需要调用外部API。"""
            }
        }
        
        # 加载环境配置
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        config_path = os.getenv('CONFIG_PATH', 'configs/dev.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                self._update_dict(self.defaults, file_config)
    
    def _update_dict(self, original: Dict, update: Dict):
        """递归更新字典"""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                self._update_dict(original[key], value)
            else:
                original[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.defaults
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# 全局配置实例
config = Config()