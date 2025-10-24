# coding:UTF-8
from aip import AipSpeech
import os
import subprocess
import platform
import tempfile
import time
import json
import getpass
import threading
import queue

# 配置文件路径
CONFIG_FILE = os.path.expanduser("~/.baidu_tts_config.json")

def load_config():
    """加载API配置信息"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    
    # 如果配置文件不存在或读取失败，提示用户输入
    print("首次使用，请输入百度语音服务的API凭证")
    app_id = input("APP_ID: ").strip()
    api_key = input("API_KEY: ").strip()
    secret_key = getpass.getpass("SECRET_KEY: ").strip()
    
    # 保存配置
    config = {
        'APP_ID': app_id,
        'API_KEY': api_key,
        'SECRET_KEY': secret_key
    }
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        # 设置文件权限为仅用户可读写
        os.chmod(CONFIG_FILE, 0o600)
        print(f"配置已保存到: {CONFIG_FILE}")
    except Exception as e:
        print(f"警告: 无法保存配置 ({str(e)}), 本次运行将使用临时配置")
    
    return config

# 加载配置
config = load_config()
APP_ID = config['APP_ID']
API_KEY = config['API_KEY']
SECRET_KEY = config['SECRET_KEY']

# 初始化客户端
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 创建播放队列和播放线程
audio_queue = queue.Queue()
playback_thread = None

def init_playback_thread():
    """初始化播放线程"""
    global playback_thread
    if playback_thread is None or not playback_thread.is_alive():
        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()

def playback_worker():
    """播放工作线程，处理音频队列"""
    while True:
        file_path = audio_queue.get()
        if file_path is None:  # 退出信号
            break
            
        try:
            # 实际播放音频
            play_audio(file_path)
            
            # 等待播放完成（估计时间）
            time.sleep(0.5)  # 确保播放开始
            while is_playing(file_path):
                time.sleep(0.1)
                
        except Exception as e:
            print(f"播放失败: {str(e)}")
        finally:
            # 清理临时文件
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print("已清理临时文件")
            except Exception as e:
                print(f"清理临时文件失败: {str(e)}")
                
        audio_queue.task_done()

def is_playing(file_path):
    """检查文件是否正在被播放（Windows专用）"""
    if platform.system() != 'Windows':
        return False  # 其他平台暂时不实现
    
    try:
        # 使用Windows API检查文件是否被占用
        import ctypes
        GENERIC_WRITE = 0x40000000
        handle = ctypes.windll.kernel32.CreateFileW(
            file_path, GENERIC_WRITE, 0, None, 3, 0, None
        )
        if handle == -1:  # INVALID_HANDLE_VALUE
            return True  # 文件被占用，可能正在播放
        ctypes.windll.kernel32.CloseHandle(handle)
        return False
    except:
        return False  # 无法检查时默认返回未播放

def text_to_speech(text_to_speak):
    """
    将文本转换为语音并加入播放队列
    :param text_to_speak: 需要转换为语音的文本
    :return: 成功返回True，失败返回False
    """
    # 检查输入文本是否为空
    if not text_to_speak.strip():
        print("警告：输入文本为空")
        return False
    
    # 检查文本长度（百度API限制1024字节）
    if len(text_to_speak.encode('utf-8')) > 1024:
        print("错误：文本长度超过1024字节限制")
        return False
    
    try:
        # 调用百度语音合成API
        result = client.synthesis(
            text_to_speak, 
            'zh', 
            1, 
            {'vol': 5, 'per': 103}
        )
        
        # 检查返回结果
        if isinstance(result, dict):
            print(f"语音合成失败: {result.get('err_msg', '未知错误')}")
            return False
            
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(result)
        
        print("语音合成成功，已加入播放队列")
        
        # 确保播放线程已启动
        init_playback_thread()
        
        # 将音频文件加入播放队列
        audio_queue.put(temp_path)
        return True
        
    except Exception as e:
        print(f"语音合成异常: {str(e)}")
        return False

def play_audio(file_path):
    """播放音频文件（Windows专用优化）"""
    system = platform.system()
    try:
        if system == 'Windows':
            # 使用Windows Media Player命令行工具直接播放
            try:
                subprocess.run(['wmplayer', file_path], check=True)
                return True
            except FileNotFoundError:
                # 如果wmplayer不可用，使用start命令
                subprocess.Popen(['start', file_path], shell=True)
                return True
        elif system == 'Darwin':  # macOS
            subprocess.run(['afplay', file_path], check=True)
            return True
        else:  # Linux
            subprocess.run(['aplay', file_path], check=True)
            return True
    except Exception as e:
        print(f"系统播放失败: {str(e)}")
        # 尝试使用playsound库
        try:
            import playsound
            playsound.playsound(file_path, block=True)
            return True
        except ImportError:
            print("请安装playsound库: pip install playsound")
        except Exception as e:
            print(f"playsound播放失败: {str(e)}")
        return False

def wait_until_done():
    """等待所有音频播放完成"""
    audio_queue.join()

# 测试示例
if __name__ == '__main__':
    # 示例1：直接调用
    text_to_speech("你好，欢迎使用语音合成服务！")
    
    # 示例2：连续调用
    text_to_speech("这是第一条消息")
    text_to_speech("这是第二条消息")
    text_to_speech("这是第三条消息")
    
    # 等待所有播放完成
    wait_until_done()
    print("所有播放任务完成")