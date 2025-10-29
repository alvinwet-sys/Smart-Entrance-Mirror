#!/usr/bin/env python3
"""
LLM接口主程序
处理core.decision_request事件，生成llm.decision_ready回复
支持日常对话、日期时间、天气、空气质量、新闻查询功能
"""


import json
import time
import uuid
import logging
import re
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError

from .config import config
from .api_handlers import APIHandlers

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMInterface:
    """LLM接口主类"""
    
    def __init__(self):
        self.api_handlers = APIHandlers()
        
        # 加载Schema
        self.decision_ready_schema = self._load_schema('decision_ready.json')
        self.error_event_schema = self._load_schema('error_event.json')
        
        # 初始化事件处理器
        self.setup_event_handlers()
    
    def _load_schema(self, schema_file: str) -> Dict[str, Any]:
        """加载JSON Schema"""
        try:
            schema_path = f"./voice_llm/llm_worker/contracts/schemas/{schema_file}"
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载Schema失败: {str(e)}")
            raise
    
    def setup_event_handlers(self):
        """设置事件处理器（在实际项目中这里会连接到事件总线）"""
        # 在实际实现中，这里会订阅core.decision_request事件
        # 并设置回调函数为self.handle_decision_request
        pass
    
    def handle_decision_request(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理决策请求
        返回decision_ready事件或error_event
        """
        try:
            # 验证输入事件格式
            self._validate_input(event_data)
            
            # 提取用户查询
            user_query = event_data.get('query', '')
            trace_id = event_data.get('trace_id', '')
            context = event_data.get('context', {})
            
            logger.info(f"处理用户查询: {user_query}, trace_id: {trace_id}")
            
            # 分析用户意图并生成回复
            reply_result = self._generate_reply(user_query, context, trace_id)
            
            if reply_result['success']:
                return self._create_decision_ready_event(
                    trace_id=trace_id,
                    reply_text=reply_result['content'],
                    actions=reply_result.get('actions', [])
                )
            else:
                return self._create_error_event(
                    trace_id=trace_id,
                    error_code="LLM_PROCESS_ERROR",
                    message=reply_result['error']
                )
                
        except Exception as e:
            logger.error(f"处理决策请求时发生错误: {str(e)}")
            return self._create_error_event(
                trace_id=event_data.get('trace_id', ''),
                error_code="LLM_SYSTEM_ERROR",
                message=f"LLM系统错误: {str(e)}"
            )
    
    def _validate_input(self, event_data: Dict[str, Any]):
        """验证输入事件数据 - 简化版本"""
    # 只检查必需字段是否存在，不做过多的格式验证
    required_fields = ['ts', 'trace_id', 'query']
    for field in required_fields:
        """
        def _validate_input(self, event_data: Dict[str, Any]):
            # 验证输入事件数据 - 简化版本 只检查必需字段是否存在，不做过多的格式验证
            required_fields = ['ts', 'trace_id', 'query']
            for field in required_fields:
                if field not in event_data:
                    raise ValueError(f"缺少必要字段: {field}")

            # 简单类型/格式检查（可根据需要调整）
            if not isinstance(event_data.get('ts'), (int, float)):
                raise ValueError("字段 ts 必须为数字时间戳")
            trace_id = event_data.get('trace_id')
            if not isinstance(trace_id, str) or len(trace_id) < 8:
                raise ValueError("字段 trace_id 必须为字符串且长度合理")
        """
    def _generate_reply(self, user_query: str, context: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """生成回复内容"""
        
        # 检查是否需要调用外部API
        api_result = self._check_and_call_apis(user_query)
        if api_result['need_api']:
            # 使用API结果生成回复
            return self._generate_reply_with_api(user_query, api_result, context, trace_id)
        else:
            # 直接使用LLM生成回复
            return self._generate_reply_directly(user_query, context, trace_id)
    
    def _check_and_call_apis(self, user_query: str) -> Dict[str, Any]:
        """检查是否需要调用API并执行调用"""
        query_lower = user_query.lower()
        
        # 检查天气查询
        if any(keyword in query_lower for keyword in ['天气', 'weather']):
            city = self._extract_city(user_query)
            if city:
                return {
                    'need_api': True,
                    'api_type': 'weather',
                    'city': city,
                    'result': self.api_handlers.handle_weather_query(city)
                }
        
        # 检查空气质量查询
        if any(keyword in query_lower for keyword in ['空气质量', '空气', 'pm2.5', 'aqi']):
            city = self._extract_city(user_query)
            return {
                'need_api': True,
                'api_type': 'air_quality',
                'city': city,
                'result': self.api_handlers.handle_air_quality_query(city)
            }
        
        # 检查新闻查询
        if any(keyword in query_lower for keyword in ['新闻', '头条', 'news']):
            news_type = self._extract_news_type(user_query)
            return {
                'need_api': True,
                'api_type': 'news',
                'news_type': news_type,
                'result': self.api_handlers.handle_news_query(news_type)
            }
        
        # 检查日期时间查询
        if any(keyword in query_lower for keyword in ['时间', '日期', '几点', '今天', '现在']):
            return {
                'need_api': True,
                'api_type': 'datetime',
                'result': self.api_handlers.get_current_datetime()
            }
        
        return {'need_api': False}
    
    def _extract_city(self, query: str) -> str:
        """从查询中提取城市名称"""
        # 简单的城市提取逻辑，可以根据需要增强
        city_patterns = [
            r'(.+?)的天气',
            r'查询(.+?)的天气',
            r'(.+?)天气',
            r'天气(.+?)'
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, query)
            if match:
                city = match.group(1).strip()
                if city and len(city) < 10:  # 简单的城市名称验证
                    return city
        
        # 如果没有明确城市，返回空字符串或默认城市
        return "北京"  # 默认城市
    
    def _extract_news_type(self, query: str) -> str:
        """从查询中提取新闻类型"""
        # 可以根据需要添加更多新闻类型映射
        type_mapping = {
            '体育': 'tiyu',
            '科技': 'keji',
            '娱乐': 'yule',
            '军事': 'junshi',
            '教育': 'jiaoyu',
            '财经': 'caijing',
            '健康': 'jiankang'
        }
        
        query_lower = query.lower()
        for chinese_type, english_type in type_mapping.items():
            if chinese_type in query_lower:
                return english_type
        
        return ""  # 空字符串表示获取全部新闻
    
    def _generate_reply_with_api(self, user_query: str, api_result: Dict[str, Any], 
                               context: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """使用API结果生成回复"""
        try:
            # 构建LLM提示词
            messages = [
                {
                    "role": "system",
                    "content": config.get('prompts.system_prompt')
                },
                {
                    "role": "user",
                    "content": f"用户查询: {user_query}\n\nAPI返回数据: {api_result['result']}\n\n请根据API返回的数据和用户查询，生成一个自然、友好的回复。"
                }
            ]
            
            llm_result = self.api_handlers.call_deepseek_api(messages)
            
            if llm_result['success']:
                return {
                    'success': True,
                    'content': llm_result['content'],
                    'actions': []
                }
            else:
                # 如果LLM调用失败，直接使用API结果
                return {
                    'success': True,
                    'content': api_result['result'],
                    'actions': []
                }
                
        except Exception as e:
            logger.error(f"使用API生成回复时出错: {str(e)}")
            return {
                'success': False,
                'error': f"生成回复失败: {str(e)}"
            }
    
    def _generate_reply_directly(self, user_query: str, context: Dict[str, Any], 
                               trace_id: str) -> Dict[str, Any]:
        """直接使用LLM生成回复"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": config.get('prompts.system_prompt')
                },
                {
                    "role": "user", 
                    "content": user_query
                }
            ]
            
            # 添加上下文信息
            identity = context.get('identity')
            if identity and identity != 'Stranger':
                messages[0]['content'] += f"\n当前用户身份: {identity}"
            
            last_asr = context.get('last_asr')
            if last_asr:
                messages[0]['content'] += f"\n上一次对话: {last_asr}"
            
            llm_result = self.api_handlers.call_deepseek_api(messages)
            
            if llm_result['success']:
                return {
                    'success': True,
                    'content': llm_result['content'],
                    'actions': []
                }
            else:
                return {
                    'success': False,
                    'error': llm_result['error']
                }
                
        except Exception as e:
            logger.error(f"直接生成回复时出错: {str(e)}")
            return {
                'success': False,
                'error': f"LLM处理失败: {str(e)}"
            }
    
    def _create_decision_ready_event(self, trace_id: str, reply_text: str, 
                                   actions: list = None) -> Dict[str, Any]:
        """创建decision_ready事件"""
        event = {
            "ts": time.time(),
            "trace_id": trace_id,
            "source": "llm",
            "reply_text": reply_text[:512],  # 确保不超过最大长度
            "actions": actions or [],
            "priority": config.get('priorities.llm', 7)
        }
        
        # 验证事件格式
        try:
            validate(instance=event, schema=self.decision_ready_schema)
            return event
        except ValidationError as e:
            logger.error(f"DecisionReady事件格式验证失败: {e.message}")
            # 返回错误事件
            return self._create_error_event(
                trace_id=trace_id,
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"DecisionReady事件格式错误: {e.message}"
            )
    
    def _create_error_event(self, trace_id: str, error_code: str, 
                          message: str, retry_after_ms: int = 0) -> Dict[str, Any]:
        """创建错误事件"""
        event = {
            "ts": time.time(),
            "trace_id": trace_id,
            "source": "llm",
            "error_code": error_code,
            "message": message,
            "retry_after_ms": retry_after_ms
        }
        
        try:
            validate(instance=event, schema=self.error_event_schema)
            return event
        except ValidationError as e:
            logger.error(f"ErrorEvent事件格式验证失败: {e.message}")
            # 如果连错误事件都验证失败，返回一个最简格式
            return {
                "ts": time.time(),
                "trace_id": trace_id,
                "source": "llm",
                "error_code": "SCHEMA_VALIDATION_ERROR",
                "message": f"原始错误: {message}, Schema错误: {e.message}",
                "retry_after_ms": 0
            }


def main():
    """主函数 - 测试用"""
    llm_interface = LLMInterface()
    
    # 完整的测试用例
    test_cases = [
        {
            "ts": time.time(),
            "trace_id": "a1b2c3d4e5f6789012345678901234567",  # 32位hex
            "source": "core",
            "query": "今天天气怎么样？",
            "context": {"identity": "Alice"}
        },
        {
            "ts": time.time(),
            "trace_id": "b2c3d4e5f67890123456789012345678",
            "source": "core", 
            "query": "北京空气质量如何？",
            "context": {"identity": "Bob"}
        },
        {
            "ts": time.time(),
            "trace_id": "c3d4e5f6789012345678901234567890",
            "source": "core",
            "query": "有什么新闻？",
            "context": {}
        },
        {
            "ts": time.time(),
            "trace_id": "d4e5f678901234567890123456789012", 
            "source": "core",
            "query": "现在几点了？",
            "context": {"last_asr": "现在时间"}
        },
        {
            "ts": time.time(),
            "trace_id": "e5f67890123456789012345678901234",
            "source": "core",
            "query": "你好，介绍一下你自己",
            "context": {"identity": "Stranger"}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试查询: {test_case['query']}")
        print(f"Trace ID: {test_case['trace_id']}")
        
        result = llm_interface.handle_decision_request(test_case)
        
        if 'reply_text' in result:
            print(f"✅ 成功生成回复: {result['reply_text'][:100]}...")
            print(f"   优先级: {result.get('priority', 7)}")
        elif 'error_code' in result:
            print(f"❌ 生成错误: {result['error_code']} - {result['message']}")
        else:
            print(f"⚠️ 未知结果: {result}")

if __name__ == "__main__":
    main()