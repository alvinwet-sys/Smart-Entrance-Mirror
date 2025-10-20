#!/usr/bin/env python3
"""
LLM模块测试
"""

import asyncio
import json
import time
import uuid
from llm_interface import LLMInterface

async def test_llm_interface():
    """测试LLM接口"""
    print("测试LLM接口...")
    
    llm = LLMInterface()
    
    # 测试数据
    test_cases = [
        {
            "query": "今天天气怎么样？",
            "context": {"identity": "Alice", "last_asr": "你好小镜"}
        },
        {
            "query": "现在几点了？",
            "context": {"identity": "Bob", "last_asr": ""}
        },
        {
            "query": "路况如何？",
            "context": {"identity": "Stranger", "last_asr": "查询路况"}
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1} ---")
        
        event_data = {
            "ts": time.time(),
            "trace_id": uuid.uuid4().hex,
            "source": "core",
            "query": test_case["query"],
            "context": test_case["context"]
        }
        
        try:
            result = await llm.process_decision_request(event_data)
            print("✓ 处理成功")
            print(f"响应: {result['reply_text']}")
            if result.get('actions'):
                print(f"动作: {result['actions']}")
                
        except Exception as e:
            print(f"✗ 处理失败: {e}")

async def test_health_check():
    """测试健康检查"""
    print("\n测试健康检查...")
    llm = LLMInterface()
    healthy = await llm.health_check()
    print(f"健康状态: {'✓ 正常' if healthy else '✗ 异常'}")

if __name__ == "__main__":
    print("开始LLM模块测试...")
    asyncio.run(test_health_check())
    asyncio.run(test_llm_interface())
    print("\n测试完成！")