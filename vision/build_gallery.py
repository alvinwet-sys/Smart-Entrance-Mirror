# rebuild_gallery.py
import os
import cv2
import numpy as np
import json
import onnxruntime as ort
from pathlib import Path

def rebuild_gallery_with_buffalo(model_path, gallery_dir, output_file=None):
    """
    使用Buffalo_M模型重新构建注册库
    
    参数:
        model_path: Buffalo_M模型文件路径
        gallery_dir: 注册库目录路径
        output_file: 输出文件路径（可选，默认为gallery_dir/gallery_data_buffalo.json）
    
    返回:
        bool: 成功返回True，失败返回False
    """
    # 设置输出文件路径
    if output_file is None:
        output_file = Path(gallery_dir) / "gallery_data_buffalo.json"
    else:
        output_file = Path(output_file)
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: Buffalo_M模型文件不存在: {model_path}")
        return False
    
    # 检查注册库目录是否存在
    if not os.path.exists(gallery_dir):
        print(f"错误: 注册库目录不存在: {gallery_dir}")
        return False
    
    # 加载Buffalo_M模型
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        recognizer = ort.InferenceSession(model_path, providers=providers)
        print(f"已加载Buffalo_M模型: {model_path}")
        
        # 打印模型信息
        print("模型输入信息:")
        for input_info in recognizer.get_inputs():
            print(f"  - 名称: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
        
    except Exception as e:
        print(f"加载Buffalo_M模型失败: {e}")
        return False
    
    # 加载人脸检测器
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except Exception as e:
        print(f"加载人脸检测器失败: {e}")
        return False
    
    # 创建新的注册库
    gallery_embeddings = {}
    
    # 遍历所有子目录
    for person_name in os.listdir(gallery_dir):
        person_dir = os.path.join(gallery_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        print(f"正在处理 {person_name} 的人脸数据...")
        features_list = []
        
        # 遍历目录中的所有图片文件
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, filename)
                
                try:
                    # 读取图像
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"无法读取图像: {image_path}")
                        continue
                    
                    print(f"处理图像: {filename}, 尺寸: {image.shape}")
                    
                    # 检测人脸
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    print(f"检测到 {len(faces)} 个人脸")
                    
                    if len(faces) == 0:
                        print(f"在 {image_path} 中未检测到人脸")
                        continue
                    
                    # 使用第一个人脸
                    x, y, w, h = faces[0]
                    
                    # 裁剪人脸区域
                    face_roi = image[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        print(f"人脸区域为空: {image_path}")
                        continue
                    
                    # 调整大小为模型期望的输入尺寸
                    aligned_face = cv2.resize(face_roi, (112, 112))
                    
                    # 使用Buffalo_M模型的预处理方式
                    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    aligned_face = aligned_face.astype(np.float32)
                    aligned_face = aligned_face / 255.0  # 归一化到[0,1]
                    aligned_face = (aligned_face - 0.5) / 0.5  # 归一化到[-1,1]
                    
                    # 调整维度
                    aligned_face = np.transpose(aligned_face, (2, 0, 1))
                    aligned_face = np.expand_dims(aligned_face, axis=0)
                    
                    # 提取特征
                    feature = recognizer.run(
                        None, 
                        {recognizer.get_inputs()[0].name: aligned_face}
                    )[0][0]
                    
                    # 归一化特征向量
                    feature_norm = np.linalg.norm(feature)
                    if feature_norm == 0:
                        print(f"特征向量范数为零: {image_path}")
                        continue
                    
                    feature = feature / feature_norm
                    
                    # 添加到特征列表
                    features_list.append(feature.tolist())
                    print(f"已提取 {person_name} 的特征 ({len(features_list)})")
                    
                except Exception as e:
                    print(f"处理 {image_path} 时出错: {e}")
                    continue
        
        if features_list:
            gallery_embeddings[person_name] = features_list
            print(f"已处理 {person_name} 的 {len(features_list)} 个面部特征")
        else:
            print(f"警告: {person_name} 没有有效的面部特征")
    
    # 保存新的注册库
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'gallery_embeddings': gallery_embeddings}, f, indent=4)
        
        print(f"新的注册库已保存到: {output_file}")
        print(f"注册库包含 {len(gallery_embeddings)} 个身份")
        
        # 打印每个身份的特征数量
        for person_name, features in gallery_embeddings.items():
            print(f"  - {person_name}: {len(features)} 个特征")
        
        return True
        
    except Exception as e:
        print(f"保存注册库失败: {e}")
        return False

if __name__ == "__main__":
    # 配置参数
    model_path = r"C:\visualprocess\w600k_r50.onnx"
    gallery_dir = r"C:\visualprocess\gallery_dataset"
    output_file = r"C:\visualprocess\gallery_dataset\gallery_data_buffalo.json"
    
    print("开始使用Buffalo_M模型重新构建注册库...")
    print(f"模型路径: {model_path}")
    print(f"注册库目录: {gallery_dir}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    # 重新构建注册库
    success = rebuild_gallery_with_buffalo(model_path, gallery_dir, output_file)
    
    if success:
        print("-" * 50)
        print("注册库重建成功!")
        print("请使用新的注册库文件运行人脸识别系统")
    else:
        print("-" * 50)
        print("注册库重建失败!")