import subprocess
import requests
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedMCPClient:
    """
    增强的MCP客户端
    自动检测MCP服务可用性，不可用时降级到直接API调用
    """
    
    def __init__(self, config=None):
        self.config = config
        self.api_keys = {
            'weather': 'd77e2863faae221ab99eec3a42b72474',
            'air_quality': 'f539234597c17a6f098be7f8bb866d09',
            'news': '0a30380cc45df2eab487d1c6305df60b'
        }
        
        # MCP服务状态
        self.mcp_available = False
        self.mcp_process = None
        self.last_mcp_check = 0
        self.mcp_check_interval = 30  # 30秒检查一次
        
        # 初始检测MCP服务
        self._check_mcp_availability()
    
    def _check_mcp_availability(self) -> bool:
        """检查MCP服务是否可用"""
        current_time = time.time()
        
        # 如果最近检查过且间隔不够，直接返回上次结果
        if current_time - self.last_mcp_check < self.mcp_check_interval:
            return self.mcp_available
        
        self.last_mcp_check = current_time
        
        try:
            # 尝试启动MCP代理服务
            if self.config:
                mcp_config = self.config.get('apis.juhe_mcp', {})
                command = mcp_config.get('command', 'uvx')
                args = mcp_config.get('args', [])
                
                if command and args:
                    logger.info("🔍 尝试启动MCP代理服务...")
                    # 检查命令是否存在
                    result = subprocess.run([command, '--version'], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=5)
                    if result.returncode == 0:
                        logger.info("✅ MCP命令可用，但未实现完整连接逻辑")
                        self.mcp_available = False  # 暂时设为False，因为还未实现完整MCP客户端
                    else:
                        logger.warning("⚠️ MCP命令不可用")
                        self.mcp_available = False
                else:
                    logger.warning("⚠️ MCP配置不完整")
                    self.mcp_available = False
            else:
                logger.warning("⚠️ 没有配置信息")
                self.mcp_available = False
                
        except Exception as e:
            logger.warning(f"⚠️ MCP服务检测失败: {e}")
            self.mcp_available = False
        
        if not self.mcp_available:
            logger.info("📡 MCP服务不可用，将使用直接API调用")
        
        return self.mcp_available
    
    def call_weather_api(self, city: str) -> Dict[str, Any]:
        """调用天气API - 优先使用MCP，降级到直接调用"""
        if self._check_mcp_availability():
            return self._call_weather_via_mcp(city)
        else:
            return self._call_weather_direct(city)
    
    def call_air_quality_api(self, city: str = "") -> Dict[str, Any]:
        """调用空气质量API"""
        if self._check_mcp_availability():
            return self._call_air_quality_via_mcp(city)
        else:
            return self._call_air_quality_direct(city)
    
    def call_news_api(self, news_type: str = "") -> Dict[str, Any]:
        """调用新闻API"""
        if self._check_mcp_availability():
            return self._call_news_via_mcp(news_type)
        else:
            return self._call_news_direct(news_type)
    
    # MCP调用方法（占位符，需要完整实现）
    def _call_weather_via_mcp(self, city: str) -> Dict[str, Any]:
        """通过MCP调用天气API"""
        logger.info(f"🌐 [MCP] 查询天气: {city}")
        # TODO: 实现真正的MCP调用
        return {'error': 'MCP调用未完全实现', 'reason': '功能开发中'}
    
    def _call_air_quality_via_mcp(self, city: str) -> Dict[str, Any]:
        """通过MCP调用空气质量API"""
        logger.info(f"🌐 [MCP] 查询空气质量: {city}")
        # TODO: 实现真正的MCP调用
        return {'error': 'MCP调用未完全实现', 'reason': '功能开发中'}
    
    def _call_news_via_mcp(self, news_type: str) -> Dict[str, Any]:
        """通过MCP调用新闻API"""
        logger.info(f"🌐 [MCP] 查询新闻: {news_type}")
        # TODO: 实现真正的MCP调用
        return {'error': 'MCP调用未完全实现', 'reason': '功能开发中'}
    
    # 直接API调用方法（当前工作的版本）
    def _call_weather_direct(self, city: str) -> Dict[str, Any]:
        """直接调用天气API"""
        try:
            logger.info(f"📡 [直接API] 查询天气: {city}")
            url = 'http://apis.juhe.cn/simpleWeather/query'
            params = {
                'key': self.api_keys['weather'],
                'city': city
            }
            
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            
            # 添加调用方式标记
            result['_call_method'] = 'direct_api'
            return result
                
        except Exception as e:
            logger.error(f"❌ 天气API直接调用失败: {str(e)}")
            return {'error': str(e), 'reason': 'API调用失败', '_call_method': 'direct_api'}
    
    def _call_air_quality_direct(self, city: str = "") -> Dict[str, Any]:
        """直接调用空气质量API"""
        try:
            # 城市名称到城市ID的映射表
            city_id_map = {
                '北京': '1',
                '上海': '2', 
                '天津': '3',
                '重庆': '4',
                '广州': '5',
                '深圳': '6',
                '杭州': '7',
                '南京': '8',
                '武汉': '9',
                '成都': '10',
                '西安': '11',
                '沈阳': '12',
                '青岛': '13',
                '大连': '14',
                '厦门': '15',
                '苏州': '16',
                '济南': '17',
                '长沙': '18',
                '郑州': '19',
                '石家庄': '20'
            }
            
            # 获取城市ID
            city_id = city_id_map.get(city, '') if city else ''
            
            logger.info(f"📡 [直接API] 查询空气质量: {city or '全国'} (ID: {city_id or '空'})")
            url = 'http://apis.juhe.cn/fapigw/air/live'
            params = {
                'key': self.api_keys['air_quality'],
                'cityId': city_id
            }
            
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            result['_call_method'] = 'direct_api'
            return result
                
        except Exception as e:
            logger.error(f"❌ 空气质量API直接调用失败: {str(e)}")
            return {'error': str(e), 'reason': 'API调用失败', '_call_method': 'direct_api'}
    
    def _call_news_direct(self, news_type: str = "") -> Dict[str, Any]:
        """直接调用新闻API"""
        try:
            logger.info(f"📡 [直接API] 查询新闻: {news_type or '全部'}")
            url = 'http://v.juhe.cn/toutiao/index'
            params = {
                'key': self.api_keys['news'],
                'type': news_type
            }
            
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            result['_call_method'] = 'direct_api'
            return result
                
        except Exception as e:
            logger.error(f"❌ 新闻API直接调用失败: {str(e)}")
            return {'error': str(e), 'reason': 'API调用失败', '_call_method': 'direct_api'}
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'mcp_available': self.mcp_available,
            'last_mcp_check': self.last_mcp_check,
            'api_keys_configured': len(self.api_keys),
            'fallback_mode': not self.mcp_available
        }
    
    def __del__(self):
        """清理资源"""
        if self.mcp_process:
            try:
                self.mcp_process.terminate()
            except:
                pass


# 测试函数
def test_enhanced_mcp_client():
    """测试增强的MCP客户端"""
    print("🧪 测试增强的MCP客户端")
    print("=" * 40)
    
    # 模拟配置
    test_config = {
        'apis': {
            'juhe_mcp': {
                'command': 'uvx',
                'args': [
                    'juhe-mcp-proxy',
                    'https://mcp.juhe.cn/sse?token=r6fdf23P0ZkhlX1RDulv9AIwqPkoVPRtkUOugd1qqwhV5Y'
                ]
            }
        }
    }
    
    class MockConfig:
        def get(self, key, default=None):
            keys = key.split('.')
            value = test_config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    
    client = EnhancedMCPClient(config=MockConfig())
    
    # 显示服务状态
    status = client.get_service_status()
    print(f"📊 服务状态: {status}")
    
    # 测试各种API调用
    test_cases = [
        ("天气查询", "北京"),
        ("空气质量", ""),
        ("新闻查询", "")
    ]
    
    for test_name, param in test_cases:
        print(f"\n🔍 测试 {test_name}: {param or '默认'}")
        
        if "天气" in test_name:
            result = client.call_weather_api(param)
        elif "空气" in test_name:
            result = client.call_air_quality_api(param)
        elif "新闻" in test_name:
            result = client.call_news_api(param)
        
        call_method = result.get('_call_method', '未知')
        print(f"   调用方式: {call_method}")
        
        if 'error' in result:
            print(f"   ❌ 错误: {result['error']}")
        elif result.get('reason') == '查询成功!' or result.get('reason') == '查询成功':
            print(f"   ✅ 成功: {result.get('reason')}")
        else:
            print(f"   ⚠️ 其他结果: {result.get('reason', '未知状态')}")

if __name__ == "__main__":
    test_enhanced_mcp_client()