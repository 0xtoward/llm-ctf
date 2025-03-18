from deepface import DeepFace
import cv2
import time
from collections import deque
import numpy as np

def process_video(video_path, sample_interval=30):
    """视频处理函数，返回关键帧的人脸特征向量"""
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % sample_interval == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 提取人脸特征向量（返回字典列表）
                embeddings = DeepFace.represent(
                    img_path=rgb_frame,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    enforce_detection=True,
                    align=True
                )
                # 提取第一个检测人脸的embedding向量
                if embeddings:
                    features.append(embeddings[0]['embedding'])  # 关键修复点
            except Exception as e:
                print(f"视频{video_path}第{frame_count}帧检测失败: {str(e)}")
        
        frame_count += 1
    
    cap.release()
    return features

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def video_verify(video1_path, video2_path, threshold=0.6):
    """优化后的视频验证函数"""
    # 提取特征向量
    print("正在处理第一个视频...")
    video1_features = process_video(video1_path)
    print(f"→ 提取到{len(video1_features)}个有效人脸特征")
    
    print("正在处理第二个视频...")
    video2_features = process_video(video2_path)
    print(f"→ 提取到{len(video2_features)}个有效人脸特征")
    
    # 特征比对优化
    match_count = 0
    total_pairs = 0
    window = deque(maxlen=10)
    
    # 限制最大比对对数（100对）
    max_pairs = min(100, len(video1_features)*len(video2_features))
    sampled_features1 = video1_features[:min(20, len(video1_features))]
    sampled_features2 = video2_features[:min(20, len(video2_features))]
    
    for feat1 in sampled_features1:
        for feat2 in sampled_features2:
            similarity = cosine_similarity(feat1, feat2)
            is_match = similarity > threshold
            
            if is_match:
                match_count +=1
                window.append(1)
            else:
                window.append(0)
            total_pairs +=1
            
            # 进度显示
            current_rate = sum(window)/len(window) if window else 0
            print(f"\r比对进度: {total_pairs}/{max_pairs} | 实时匹配率: {current_rate:.2%}", end="")
    
    # 最终判断
    match_rate = match_count / total_pairs if total_pairs >0 else 0
    final_verdict = match_rate > 0.65  # 调整阈值
    
    print(f"\n综合匹配率：{match_rate:.2%}")
    return {"verified": final_verdict, "confidence": match_rate}

# 使用示例
result = video_verify("video1.mp4", "video2.mp4")
print(f"验证结果：{'同一人' if result['verified'] else '不同人'}（置信度：{result['confidence']:.2%}）")