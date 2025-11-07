#!/usr/bin/env python3
"""
ASR参数调整工具 - 方便用户实时调整语音检测参数
"""

def show_current_config():
    """显示当前ASR配置"""
    print("🎛️ 当前ASR参数配置:")
    print("=" * 60)
    
    # 直接定义配置，避免导入整个模块
    try:
        # 从core_asr_ts.py中提取的当前配置
        asr_config = {
            'model_dir': r"C:/Users/17321/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
            
            # 🎛️ 语音检测参数 (可调整)
            'amplitude_threshold': 0.03,        # 音频信号强度阈值 (降低，更敏感)
            'silence_threshold': 0.015,         # 静音检测阈值 (降低)  
            'confidence_required': 2,           # 连续检测次数要求 (从3降低到2)
            'confidence_decay': 1,              # 置信度衰减速度
            'endpoint_silence_ms': 1500,        # 语音结束静音时长 (从2000ms减少到1500ms)
            
            # 🔍 调试和日志参数
            'debug_audio_threshold': 0.005,     # 调试日志的音频阈值
            'debug_silence_interval': 300,      # 静音累积调试输出间隔(ms)
            
            # 🚀 高级参数
            'enable_pre_detection_recording': True, # 启用语音检测前的预录音
            'pre_speech_buffer_ms': 500,        # 语音前缓冲时间(ms) - 减少丢失
        }
        
        print(f"📊 语音检测参数:")
        print(f"  • 音频信号强度阈值: {asr_config['amplitude_threshold']}")
        print(f"  • 静音检测阈值: {asr_config['silence_threshold']}")
        print(f"  • 连续检测次数要求: {asr_config['confidence_required']}")
        print(f"  • 置信度衰减速度: {asr_config['confidence_decay']}")
        print(f"  • 语音结束静音时长: {asr_config['endpoint_silence_ms']}ms")
        
        print(f"\n🔍 调试参数:")
        print(f"  • 调试日志音频阈值: {asr_config['debug_audio_threshold']}")
        print(f"  • 静音累积调试间隔: {asr_config['debug_silence_interval']}ms")
        
        print(f"\n🚀 高级参数:")
        print(f"  • 启用预录音: {asr_config['enable_pre_detection_recording']}")
        print(f"  • 语音前缓冲时间: {asr_config['pre_speech_buffer_ms']}ms")
        
        print(f"\n📁 模型路径:")
        print(f"  • 本地模型目录: {asr_config['model_dir']}")
        
        print(f"\n📝 配置文件位置:")
        print(f"  • 主配置文件: core_asr_ts.py -> AppConfig.asr")
        
    except Exception as e:
        print(f"❌ 无法读取配置: {e}")
        import traceback
        traceback.print_exc()

def generate_custom_config():
    """生成自定义配置文件"""
    print("\n🔧 ASR参数调整建议:")
    print("=" * 60)
    
    configs = {
        "敏感模式 (适合安静环境)": {
            'amplitude_threshold': 0.02,
            'silence_threshold': 0.01,
            'confidence_required': 1,
            'endpoint_silence_ms': 1000,
            'debug_audio_threshold': 0.003,
        },
        "标准模式 (推荐设置)": {
            'amplitude_threshold': 0.03,
            'silence_threshold': 0.015,
            'confidence_required': 2,
            'endpoint_silence_ms': 1500,
            'debug_audio_threshold': 0.005,
        },
        "保守模式 (适合嘈杂环境)": {
            'amplitude_threshold': 0.05,
            'silence_threshold': 0.02,
            'confidence_required': 3,
            'endpoint_silence_ms': 2000,
            'debug_audio_threshold': 0.01,
        }
    }
    
    for mode_name, params in configs.items():
        print(f"\n📋 {mode_name}:")
        for key, value in params.items():
            print(f"  • {key}: {value}")
    
    print(f"\n💡 参数说明:")
    print(f"  • amplitude_threshold: 数值越小越敏感，更容易检测到语音")
    print(f"  • silence_threshold: 数值越小对静音检测越严格")
    print(f"  • confidence_required: 数值越小响应越快，但可能误触发")
    print(f"  • endpoint_silence_ms: 数值越小语音结束检测越快")

def create_config_template():
    """创建配置模板文件"""
    template = '''#!/usr/bin/env python3
"""
自定义ASR配置文件 - 复制到core_asr_ts.py中的AppConfig类
"""

# 将以下配置复制到 core_asr_ts.py 中的 AppConfig.asr 字典中
CUSTOM_ASR_CONFIG = {
    'model_dir': r"C:/Users/17321/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    
    # 🎛️ 语音检测参数 (可调整)
    'amplitude_threshold': 0.025,       # 🔧 调整这个: 音频信号强度阈值 (0.01-0.1)
    'silence_threshold': 0.012,         # 🔧 调整这个: 静音检测阈值 (0.005-0.05)  
    'confidence_required': 2,           # 🔧 调整这个: 连续检测次数要求 (1-5)
    'confidence_decay': 1,              # 置信度衰减速度
    'endpoint_silence_ms': 1200,        # 🔧 调整这个: 语音结束静音时长 (500-3000ms)
    
    # 🔍 调试和日志参数
    'debug_audio_threshold': 0.003,     # 调试日志的音频阈值
    'debug_silence_interval': 250,      # 静音累积调试输出间隔(ms)
    
    # 🚀 高级参数
    'enable_pre_detection_recording': True, # 启用语音检测前的预录音
    'pre_speech_buffer_ms': 400,        # 语音前缓冲时间(ms) - 减少丢失
}

# 🚀 快速应用配置的步骤:
# 1. 复制上面的 CUSTOM_ASR_CONFIG 内容
# 2. 替换 core_asr_ts.py 中 AppConfig.asr 的内容  
# 3. 重启 python core_asr_ts.py
# 4. 观察日志输出，根据需要继续调整参数
'''
    
    with open('asr_config_template.py', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"\n✅ 已生成配置模板文件: asr_config_template.py")
    print(f"📝 请查看该文件获取详细的参数调整指导")

def main():
    print("🎤 ASR参数调整工具")
    print("=" * 60)
    
    while True:
        print(f"\n请选择操作:")
        print(f"1. 查看当前配置")
        print(f"2. 查看推荐配置")
        print(f"3. 生成配置模板文件")
        print(f"4. 退出")
        
        choice = input(f"\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            show_current_config()
        elif choice == '2':
            generate_custom_config()
        elif choice == '3':
            create_config_template()
        elif choice == '4':
            print(f"👋 再见！")
            break
        else:
            print(f"❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()