import json
import time
import uuid
import asyncio
import logging
from typing import Dict, List, Optional
from jsonschema import validate, Draft7Validator
import yaml
import aiohttp
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_interface")

@dataclass
class LLMConfig:
    """LLM配置类"""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    timeout: int = 8000  # 毫秒
    max_retries: int = 3
    model: str = "deepseek-chat"
    temperature: float = 0.7

class LLMInterface:
    """LLM接口处理类"""
    
    def __init__(self, config_path: str = "configs.yaml", rules_path: str = "rules.yaml"):
        self.config = self._load_config(config_path)
        self.rules = self._load_rules(rules_path)
        self.llm_config = LLMConfig(
            api_key=self.config['deepseek']['api_key'],
            timeout=self.config['timeouts']['llm_ms'],
            model=self.config['deepseek'].get('model', 'deepseek-chat')
        )
        
        # 加载schema验证器
        self.decision_request_schema = self._load_schema("decision_request")
        self.decision_ready_schema = self._load_schema("decision_ready")
        self.error_schema = self._load_schema("error_event")
        
        # API调用统计
        self.api_call_count = 0
        self.last_call_time = 0
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _load_rules(self, rules_path: str) -> Dict:
        """加载规则库"""
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载规则库失败: {e}")
            return {}
    
    def _load_schema(self, schema_name: str) -> Dict:
        """加载schema定义（简化版，实际应从文件读取）"""
        schemas = {
            "decision_request": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["ts", "trace_id", "query"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["core"]},
                    "query": {"type": "string", "minLength": 1, "maxLength": 256},
                    "context": {
                        "type": "object",
                        "properties": {
                            "identity": {"type": "string"},
                            "last_asr": {"type": "string", "minLength": 1, "maxLength": 256}
                        },
                        "additionalProperties": False
                    }
                },
                "additionalProperties": False
            },
            "decision_ready": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["ts", "trace_id", "reply_text"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["llm"]},
                    "reply_text": {"type": "string", "minLength": 1, "maxLength": 512},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 10}
                },
                "additionalProperties": False
            },
            "error_event": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["ts", "trace_id", "source", "error_code", "message"],
                "properties": {
                    "ts": {"type": "number"},
                    "trace_id": {"type": "string"},
                    "source": {"type": "string"},
                    "error_code": {"type": "string"},
                    "message": {"type": "string"},
                    "retry_after_ms": {"type": "integer", "minimum": 0}
                },
                "additionalProperties": False
            }
        }
        return schemas.get(schema_name, {})
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'deepseek': {
                'api_key': 'sk-bdd73529195f4034961136f95e61e13d',
                'model': 'deepseek-chat'
            },
            'timeouts': {
                'llm_ms': 8000
            },
            'retry': {
                'max_attempts': 3,
                'base_backoff_ms': 200
            }
        }
    
    def validate_schema(self, data: Dict, schema_type: str) -> bool:
        """验证数据是否符合schema"""
        try:
            schema = self._load_schema(schema_type)
            validator = Draft7Validator(schema)
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                logger.error(f"Schema验证失败: {[e.message for e in errors]}")
                return False
            return True
        except Exception as e:
            logger.error(f"Schema验证异常: {e}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """检查速率限制"""
        current_time = time.time()
        time_window = 60  # 60秒窗口
        
        # 简单的速率限制检查
        if current_time - self.last_call_time < 1.0:  # 1秒内只能调用1次
            return False
        
        self.last_call_time = current_time
        return True
    
    def _build_prompt(self, query: str, context: Dict) -> str:
        """构建Prompt"""
        identity = context.get('identity', '用户')
        last_asr = context.get('last_asr', '')
        
        # 系统提示词
        system_prompt = """你是一个智能家居镜的AI助手，具有以下能力：
1. 日期时间播报与计算
2. 天气预报查询
3. 路况信息
4. 日常问候与对话
5. 家居控制建议

请以自然、友好的语气回答，回答要简洁明了，适合语音播报。"""
        
        # 身份个性化
        if identity and identity != "Stranger":
            system_prompt += f"\n\n当前用户是：{identity}，请使用适合的称呼和语气。"
        
        # 构建完整Prompt
        prompt = f"{system_prompt}\n\n用户提问：{query}"
        
        if last_asr:
            prompt += f"\n上文语境：{last_asr}"
            
        return prompt
    
    async def call_deepseek_api(self, prompt: str) -> Optional[str]:
        """调用DeepSeek API"""
        if not self._check_rate_limit():
            raise Exception("API调用频率超限")
        
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": "你是一个有用的智能家居助手。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.llm_config.temperature,
            "max_tokens": 512
        }
        
        timeout = aiohttp.ClientTimeout(total=self.llm_config.timeout/1000)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.llm_config.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    elif response.status == 429:
                        raise Exception("API速率限制")
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败: {response.status} - {error_text}")
                        
        except asyncio.TimeoutError:
            raise Exception("API调用超时")
        except Exception as e:
            raise e
    
    def _extract_actions(self, response: str) -> List[str]:
        """从响应中提取动作指令"""
        actions = []
        # 简单的动作提取规则
        if "天气" in response and "查询" in response:
            actions.append("weather.query")
        if "路况" in response:
            actions.append("traffic.query")
        if "音量" in response and ("调大" in response or "增加" in response):
            actions.append("device.volume_up")
        if "音量" in response and ("调小" in response or "减小" in response):
            actions.append("device.volume_down")
        
        return actions
    
    async def process_decision_request(self, event_data: Dict) -> Dict:
        """处理决策请求"""
        # 验证输入数据
        if not self.validate_schema(event_data, "decision_request"):
            raise Exception("输入数据schema验证失败")
        
        trace_id = event_data['trace_id']
        query = event_data['query']
        context = event_data.get('context', {})
        
        logger.info(f"处理决策请求 trace_id={trace_id}, query={query}")
        
        # 构建Prompt
        prompt = self._build_prompt(query, context)
        
        # 调用LLM
        try:
            response_text = await self.call_deepseek_api(prompt)
            
            if not response_text:
                raise Exception("LLM返回空响应")
            
            # 提取动作
            actions = self._extract_actions(response_text)
            
            # 构建响应事件
            decision_ready_event = {
                "ts": time.time(),
                "trace_id": trace_id,
                "source": "llm",
                "reply_text": response_text[:500],  # 确保不超过长度限制
                "actions": actions,
                "priority": 7
            }
            
            # 验证输出数据
            if not self.validate_schema(decision_ready_event, "decision_ready"):
                raise Exception("输出数据schema验证失败")
            
            return decision_ready_event
            
        except Exception as e:
            logger.error(f"LLM处理失败: {e}")
            
            # 构建错误事件
            error_event = {
                "ts": time.time(),
                "trace_id": trace_id,
                "source": "llm",
                "error_code": "LLM_API_ERROR",
                "message": str(e),
                "retry_after_ms": 5000
            }
            
            if not self.validate_schema(error_event, "error_event"):
                logger.error("错误事件schema验证失败")
            
            raise e
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            test_prompt = "你好"
            response = await self.call_deepseek_api(test_prompt)
            return response is not None
        except:
            return False

# 使用示例
async def main():
    """测试函数"""
    llm_interface = LLMInterface()
    
    # 测试数据
    test_event = {
        "ts": time.time(),
        "trace_id": uuid.uuid4().hex,
        "source": "core",
        "query": "今天天气怎么样？",
        "context": {
            "identity": "Alice",
            "last_asr": "你好小镜"
        }
    }
    
    try:
        result = await llm_interface.process_decision_request(test_event)
        print("LLM响应:", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())