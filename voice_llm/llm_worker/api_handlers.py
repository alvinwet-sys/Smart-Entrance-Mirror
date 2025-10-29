import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .config import config
from .mcp_client import MCPClient

logger = logging.getLogger(__name__)

class APIHandlers:
    """API处理器"""
    
    def __init__(self):
        self.mcp_client = MCPClient()
        self.deepseek_config = config.get('apis.deepseek')
    
    def call_deepseek_api(self, messages: list, temperature: float = 0.7) -> Dict[str, Any]:
        """调用DeepSeek API"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.deepseek_config['api_key']}"
            }
            
            payload = {
                'model': 'deepseek-chat',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': 1024
            }
            
            response = requests.post(
                self.deepseek_config['base_url'],
                headers=headers,
                json=payload,
                timeout=config.get('timeouts.api_timeout')
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'usage': result.get('usage', {})
                }
            else:
                logger.error(f"DeepSeek API调用失败: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"API调用失败: {response.status_code}"
                }
                
        except requests.Timeout:
            logger.error("DeepSeek API调用超时")
            return {'success': False, 'error': '请求超时'}
        except Exception as e:
            logger.error(f"DeepSeek API调用异常: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def handle_weather_query(self, city: str) -> str:
        """处理天气查询"""
        result = self.mcp_client.call_weather_api(city)
        
        if 'error' in result:
            return f"抱歉，查询{city}的天气时出现错误：{result['error']}"
        
        # 解析聚合数据返回的天气信息
        if result.get('reason') == '查询成功!':
            weather_data = result.get('result', {})
            realtime = weather_data.get('realtime', {})
            
            return f"{city}当前天气：{realtime.get('info', '未知')}，温度：{realtime.get('temperature', '未知')}℃，湿度：{realtime.get('humidity', '未知')}%，风向：{realtime.get('direct', '未知')}，风力：{realtime.get('power', '未知')}级"
        else:
            return f"抱歉，未能获取到{city}的天气信息：{result.get('reason', '未知错误')}"
    
    def handle_air_quality_query(self, city: str = "") -> str:
        """处理空气质量查询"""
        result = self.mcp_client.call_air_quality_api(city)
        
        if 'error' in result:
            location = city if city else "全国"
            return f"抱歉，查询{location}空气质量时出现错误：{result['error']}"
        
        if result.get('reason') == '查询成功':
            air_data = result.get('result', [])
            if air_data:
                location = city if city else "全国"
                quality_data = air_data[0] if isinstance(air_data, list) else air_data
                
                return f"{location}空气质量：AQI指数{quality_data.get('aqi', '未知')}，空气质量{quality_data.get('quality', '未知')}，主要污染物：{quality_data.get('primary_pollutant', '无')}"
            else:
                return "抱歉，未能获取到空气质量信息"
        else:
            return f"抱歉，未能获取到空气质量信息：{result.get('reason', '未知错误')}"
    
    def handle_news_query(self, news_type: str = "") -> str:
        """处理新闻查询"""
        result = self.mcp_client.call_news_api(news_type)
        
        if 'error' in result:
            return f"抱歉，查询新闻时出现错误：{result['error']}"
        
        if result.get('reason') == '查询成功':
            news_data = result.get('result', {}).get('data', [])
            if news_data:
                news_list = []
                for i, news in enumerate(news_data[:5]):  # 只取前5条
                    title = news.get('title', '无标题')
                    author = news.get('author_name', '未知作者')
                    news_list.append(f"{i+1}. {title} - {author}")
                
                return "今日头条新闻：\n" + "\n".join(news_list)
            else:
                return "抱歉，今天没有找到相关新闻"
        else:
            return f"抱歉，未能获取到新闻信息：{result.get('reason', '未知错误')}"
    
    def get_current_datetime(self) -> str:
        """获取当前日期时间"""
        now = datetime.now()
        return now.strftime("当前日期时间：%Y年%m月%d日 %H时%M分%S秒")