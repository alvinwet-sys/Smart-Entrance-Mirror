import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MCPClient:
    """直接调用聚合数据API（简化方案）"""
    
    def __init__(self):
        self.api_keys = {
            'weather': 'd77e2863faae221ab99eec3a42b72474',
            'air_quality': 'f539234597c17a6f098be7f8bb866d09',
            'news': '0a30380cc45df2eab487d1c6305df60b'
        }
    
    def call_weather_api(self, city: str) -> Dict[str, Any]:
        """直接调用天气预报API"""
        try:
            url = 'http://apis.juhe.cn/simpleWeather/query'
            params = {
                'key': self.api_keys['weather'],
                'city': city
            }
            
            response = requests.get(url, params=params, timeout=10)
            return response.json()
                
        except Exception as e:
            logger.error(f"天气预报API调用异常: {str(e)}")
            return {'error': str(e), 'reason': 'API调用失败'}
    
    def call_air_quality_api(self, city: str = "") -> Dict[str, Any]:
        """直接调用空气质量API"""
        try:
            url = 'http://apis.juhe.cn/fapigw/air/live'
            params = {
                'key': self.api_keys['air_quality'],
                'cityId': city if city else ''
            }
            
            response = requests.get(url, params=params, timeout=10)
            return response.json()
                
        except Exception as e:
            logger.error(f"空气质量API调用异常: {str(e)}")
            return {'error': str(e), 'reason': 'API调用失败'}
    
    def call_news_api(self, news_type: str = "") -> Dict[str, Any]:
        """直接调用新闻API"""
        try:
            url = 'http://v.juhe.cn/toutiao/index'
            params = {
                'key': self.api_keys['news'],
                'type': news_type
            }
            
            response = requests.get(url, params=params, timeout=10)
            return response.json()
                
        except Exception as e:
            logger.error(f"新闻API调用异常: {str(e)}")
            return {'error': str(e), 'reason': 'API调用失败'}