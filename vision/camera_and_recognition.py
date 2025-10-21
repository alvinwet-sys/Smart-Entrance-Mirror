import os
import cv2
import numpy as np
import json
import time
import uuid
import zmq
import threading
import onnxruntime as ort
from pathlib import Path

class EventBusClient:
    """事件总线客户端封装"""
    
    def __init__(self, bus_address="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(bus_address)
        print(f"已连接到事件总线: {bus_address}")
    
    def publish_event(self, topic, event_data):
        """发布事件到总线"""
        # 确保事件包含必需字段
        if 'ts' not in event_data:
            event_data['ts'] = time.time()
        if 'trace_id' not in event_data:
            event_data['trace_id'] = str(uuid.uuid4()).replace('-', '')
        if 'source' not in event_data:
            event_data['source'] = 'vision'
        
        # 发送事件
        message = {
            'topic': topic,
            'data': event_data
        }
        self.socket.send_json(message)
        print(f"已发布事件到主题 {topic}: {json.dumps(event_data, indent=2)}")
    
    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()

class FaceTracker:
    """人脸跟踪器，用于维护人脸轨迹ID"""
    
    def __init__(self, max_disappeared=5, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # {track_id: [centroid_history, features, last_seen]}
        self.disappeared = {}  # {track_id: disappeared_count}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, features):
        """注册新的人脸轨迹"""
        object_id = self.next_object_id
        self.objects[object_id] = {
            'centroids': [centroid],
            'features': [features],
            'last_seen': time.time()
        }
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id):
        """注销人脸轨迹"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def update(self, detections):
        """更新跟踪状态"""
        # 如果没有检测到人脸，增加所有轨迹的消失计数
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # 计算当前检测框的中心点
        input_centroids = []
        input_features = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            input_centroids.append(centroid)
            input_features.append(detection.get('features', None))
        
        # 如果没有现有轨迹，注册所有检测到的人脸
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(input_centroids[i], input_features[i])
            return self.objects
        
        # 否则，尝试匹配现有轨迹
        object_ids = list(self.objects.keys())
        object_centroids = []
        for object_id in object_ids:
            centroids = self.objects[object_id]['centroids']
            object_centroids.append(centroids[-1])  # 使用最新的中心点
        
        # 计算距离矩阵
        D = np.zeros((len(object_ids), len(input_centroids)))
        for i in range(len(object_ids)):
            for j in range(len(input_centroids)):
                dx = object_centroids[i][0] - input_centroids[j][0]
                dy = object_centroids[i][1] - input_centroids[j][1]
                D[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # 匹配轨迹
        used_rows = set()
        used_cols = set()
        
        # 先匹配距离最小的
        for i in range(len(object_ids)):
            if i in used_rows:
                continue
                
            min_val = np.inf
            min_col = -1
            
            for j in range(len(input_centroids)):
                if j in used_cols:
                    continue
                if D[i, j] < min_val:
                    min_val = D[i, j]
                    min_col = j
            
            if min_val < self.max_distance:
                object_id = object_ids[i]
                self.objects[object_id]['centroids'].append(input_centroids[min_col])
                if input_features[min_col] is not None:
                    self.objects[object_id]['features'].append(input_features[min_col])
                self.objects[object_id]['last_seen'] = time.time()
                self.disappeared[object_id] = 0
                used_rows.add(i)
                used_cols.add(min_col)
        
        # 处理未匹配的轨迹和检测
        for i in range(len(object_ids)):
            if i not in used_rows:
                object_id = object_ids[i]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        for j in range(len(input_centroids)):
            if j not in used_cols:
                self.register(input_centroids[j], input_features[j])
        
        # 返回当前活跃的轨迹
        return self.objects

class SimpleFaceRecognizer:
    def __init__(self, recognition_model_path, gallery_dir, gpu_id=0):
        # 设置ONNX Runtime执行提供者
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu_id >= 0 else ['CPUExecutionProvider']
        
        # 加载人脸识别模型
        self.recognizer = ort.InferenceSession(recognition_model_path, providers=providers)
        
        # 加载人脸检测器（使用OpenCV的Haar级联分类器）
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 设置注册库目录
        self.gallery_dir = gallery_dir
        
        # 加载注册库
        if not self.load_gallery():
            raise Exception("无法加载注册库")
        
        self.threshold = 0.5
        self.tracker = FaceTracker()
        self.min_face_size = 30  # 最小人脸尺寸（像素）
        
        print("人脸识别器初始化完成")
    
    def load_gallery(self):
        """加载注册库"""
        gallery_file = Path(self.gallery_dir) / "gallery_data.json"
        
        if not gallery_file.exists():
            print(f"注册库文件不存在: {gallery_file}")
            return False
        
        try:
            with open(gallery_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.gallery_embeddings = data.get('gallery_embeddings', {})
            print(f"注册库加载成功: {len(self.gallery_embeddings)} 个身份")
            
            # 转换嵌入向量为numpy数组
            for subject_id, embeddings in self.gallery_embeddings.items():
                self.gallery_embeddings[subject_id] = [np.array(emb) for emb in embeddings]
            
            return True
            
        except Exception as e:
            print(f"加载注册库失败: {e}")
            return False
    
    def detect_faces(self, image):
        """使用OpenCV的Haar级联分类器检测图像中的人脸"""
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            results = []
            for (x, y, w, h) in faces:
                # 转换为 (x1, y1, x2, y2) 格式
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # 生成简单的关键点（基于边界框）
                landmarks = [
                    [x1 + w//3, y1 + h//3],    # 左眼
                    [x1 + 2*w//3, y1 + h//3],  # 右眼
                    [x1 + w//2, y1 + h//2],     # 鼻子
                    [x1 + w//3, y1 + 2*h//3],   # 左嘴角
                    [x1 + 2*w//3, y1 + 2*h//3]  # 右嘴角
                ]
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'landmarks': np.array(landmarks, dtype=np.float32),
                    'score': 0.9  # 默认置信度
                })
            
            return results
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            return []
    
    def extract_feature(self, image, bbox):
        """从人脸图像中提取特征向量"""
        try:
            print(f"提取特征: bbox={bbox}, image_shape={image.shape}")
            
            # 检查输入参数
            if image is None:
                print("错误: 图像为None")
                return None
            if bbox is None:
                print("错误: 边界框为None")
                return None
            
            x1, y1, x2, y2 = bbox
            print(f"边界框坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # 检查边界框是否有效
            if x1 >= x2 or y1 >= y2:
                print("错误: 无效的边界框")
                return None
            
            # 裁剪人脸区域
            face_roi = image[y1:y2, x1:x2]
            print(f"人脸区域形状: {face_roi.shape}")
            
            if face_roi.size == 0:
                print("错误: 人脸区域为空")
                return None
            
            # 调整大小为模型期望的输入尺寸
            aligned_face = cv2.resize(face_roi, (112, 112))
            print(f"调整后的人脸形状: {aligned_face.shape}")
            
            # 预处理对齐后的人脸
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            aligned_face = aligned_face.astype(np.float32)
            aligned_face = (aligned_face - 127.5) / 128.0
            
            # 检查预处理后的值
            print(f"预处理后 - 最小值: {aligned_face.min()}, 最大值: {aligned_face.max()}, 平均值: {aligned_face.mean()}")
            
            # 调整维度
            aligned_face = np.transpose(aligned_face, (2, 0, 1))
            aligned_face = np.expand_dims(aligned_face, axis=0)
            print(f"调整维度后形状: {aligned_face.shape}")
            
            # 提取特征
            feature = self.recognizer.run(
                None, 
                {self.recognizer.get_inputs()[0].name: aligned_face}
            )[0][0]
            
            print(f"原始特征形状: {feature.shape}, 范数: {np.linalg.norm(feature)}")
            
            # 归一化特征向量
            feature_norm = np.linalg.norm(feature)
            if feature_norm == 0:
                print("错误: 特征向量范数为零")
                return None
                
            feature = feature / feature_norm
            print(f"归一化后特征范数: {np.linalg.norm(feature)}")
            
            return feature
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def recognize(self, query_embedding):
        """识别人脸"""
        print(f"开始识别: query_embedding={query_embedding is not None}")
        
        # 检查查询嵌入向量是否为None
        if query_embedding is None:
            print("错误: 查询嵌入向量为None")
            return "stranger", 0.0
            
        if not self.gallery_embeddings:
            print("警告: 注册库为空")
            return "stranger", 0.0
        
        best_similarity = 0.0
        best_subject = "stranger"
        
        for subject_id, embeddings in self.gallery_embeddings.items():
            print(f"比较身份: {subject_id}, 嵌入数量: {len(embeddings)}")
            
            for i, gallery_embedding in enumerate(embeddings):
                # 检查gallery_embedding是否为None
                if gallery_embedding is None:
                    print(f"警告: {subject_id} 的第 {i} 个嵌入向量为None")
                    continue
                    
                # 计算余弦相似度
                try:
                    similarity = np.dot(query_embedding, gallery_embedding)
                    print(f"与 {subject_id} 的相似度: {similarity}")
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_subject = subject_id
                except Exception as e:
                    print(f"相似度计算失败: {e}")
                    continue
        
        print(f"最佳匹配: {best_subject}, 相似度: {best_similarity}")
        
        # 应用阈值
        if best_similarity < self.threshold:
            return "stranger", best_similarity
        else:
            return best_subject, best_similarity
    
    def process_frame(self, frame, trace_id):
        """处理单帧图像并返回识别结果"""
        # 检测人脸
        faces = self.detect_faces(frame)
        
        # 提取特征并识别
        results = []
        for face in faces:
            # 提取特征
            feature = self.extract_feature(frame, face['bbox'])
            if feature is None:
                continue
            
            # 识别人脸
            identity, similarity = self.recognize(feature)
            
            results.append({
                'bbox': face['bbox'],
                'embedding': feature,
                'identity': identity,
                'similarity': float(similarity),
                'score': float(face['score'])
            })
        
        # 更新跟踪器
        tracked_faces = self.tracker.update(results)
        
        # 准备事件数据
        events = []
        for track_id, track_data in tracked_faces.items():
            # 使用最新的特征进行识别
            if len(track_data['features']) > 0:
                latest_features = track_data['features'][-1]
                identity, similarity = self.recognize(latest_features)
                
                # 使用最新的位置
                latest_centroid = track_data['centroids'][-1]
                
                # 创建事件
                event_data = {
                    'ts': time.time(),
                    'trace_id': trace_id,
                    'source': 'vision',
                    'track_id': track_id,
                    'bbox': self.estimate_bbox_from_centroid(latest_centroid, frame.shape),
                    'identity': identity,
                    'similarity': float(similarity),
                    'quality': float(track_data.get('score', 0.5))
                }
                
                events.append(event_data)
        
        return events, results
    
    def estimate_bbox_from_centroid(self, centroid, image_shape):
        """从中心点估计边界框"""
        h, w = image_shape[:2]
        cx, cy = centroid
        bbox_size = min(w, h) * 0.2  # 假设人脸占图像的20%
        
        x1 = max(0, int(cx - bbox_size / 2))
        y1 = max(0, int(cy - bbox_size / 2))
        x2 = min(w, int(cx + bbox_size / 2))
        y2 = min(h, int(cy + bbox_size / 2))
        
        return [x1, y1, x2, y2]
    
    def draw_detections(self, frame, results):
        """在图像上绘制检测结果"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            similarity = result['similarity']
            
            # 绘制边界框
            color = (0, 255, 0) if identity != "stranger" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制身份标签
            label = f"{identity} ({similarity:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

class RealTimeFaceRecognition:
    """实时人脸识别系统"""
    
    def __init__(self, recognition_model_path, gallery_dir, bus_address="tcp://localhost:5555"):
        # 初始化事件总线
        self.event_bus = EventBusClient(bus_address)
        
        # 初始化人脸识别器
        self.recognizer = SimpleFaceRecognizer(recognition_model_path, gallery_dir)
        
        # 初始化摄像头
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("无法打开摄像头")
        
        # 设置摄像头参数
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 控制变量
        self.is_running = False
        self.processing_thread = None
        
        print("实时人脸识别系统初始化完成")
    
    def start(self):
        """启动系统"""
        if self.is_running:
            print("系统已在运行中")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("实时人脸识别系统已启动")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.camera.release()
        self.event_bus.close()
        cv2.destroyAllWindows()
        
        print("实时人脸识别系统已停止")
    
    def _processing_loop(self):
        """处理循环"""
        frame_interval = 0.1  # 处理间隔（秒）
        last_process_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            # 控制处理频率
            if current_time - last_process_time < frame_interval:
                time.sleep(0.01)
                continue
            
            # 读取帧
            ret, frame = self.camera.read()
            if not ret:
                print("无法读取摄像头帧")
                time.sleep(1.0)
                continue
            
            # 生成追踪ID
            trace_id = str(uuid.uuid4()).replace('-', '')
            
            # 处理帧
            try:
                print(f"开始处理帧 {trace_id}, 帧形状: {frame.shape}")
                
                events, results = self.recognizer.process_frame(frame, trace_id)
                print(f"处理完成: {len(events)} 个事件, {len(results)} 个结果")
                
                # 发布事件
                for event_data in events:
                    self.event_bus.publish_event("core.face_id_resolved", event_data)
                
                # 在帧上绘制结果
                self.recognizer.draw_detections(frame, results)
                
                # 显示帧
                cv2.imshow('实时人脸识别', frame)
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                
                last_process_time = current_time
                
            except Exception as e:
                print(f"处理帧时出错: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的堆栈跟踪
                time.sleep(1.0)

# 使用示例
if __name__ == "__main__":
    # 配置参数
    recognition_model_path = r"C:\visualprocess\LVFace-L_Glint360K.onnx"
    gallery_dir = r"C:\visualprocess\gallery_dataset"
    bus_address = "tcp://localhost:5555"
    
    # 初始化 recognizer 为 None
    recognizer = None
    
    try:
        # 创建并启动系统
        recognizer = RealTimeFaceRecognition(recognition_model_path, gallery_dir, bus_address)
        
        # 启动系统
        recognizer.start()
        
        # 等待用户输入停止
        print("系统运行中，按Enter键停止...")
        input()
        
    except Exception as e:
        print(f"启动系统失败: {e}")
    
    finally:
        # 只有在 recognizer 已初始化时才调用 stop()
        if recognizer is not None:
            recognizer.stop()
        else:
            print("系统未成功初始化，无需停止")