# test_tts.py
from tts_module import text_to_speech, wait_until_done
import time

def main():
    print("开始测试文本转语音功能")
    
    # 测试1: 简单问候
    print("\n测试1: 简单问候")
    text_to_speech("你好，欢迎使用文本转语音服务！")
    
    # 等待播放完成
    time.sleep(2)  # 给播放一些时间
    
    # 测试2: 多个连续消息
    print("\n测试2: 多个连续消息")
    messages = [
        "这是第一条消息",
        "这是第二条消息",
        "这是第三条消息"
    ]
    for msg in messages:
        text_to_speech(msg)
        time.sleep(1)  # 间隔1秒
    
    # 测试3: 长文本（自动分割）
    print("\n测试3: 长文本（自动分割）")
    long_text = ("这是一段很长的文本，用于测试文本分割功能。" * 20) + "文本结束。"
    text_to_speech(long_text)
    
    # 测试4: 特殊字符
    print("\n测试4: 特殊字符")
    text_to_speech("特殊字符测试：~!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./")
    
    # 测试5: 空文本
    print("\n测试5: 空文本")
    text_to_speech("")
    
    # 等待所有音频播放完成
    wait_until_done()
    print("\n所有测试完成！")

if __name__ == '__main__':
    main()