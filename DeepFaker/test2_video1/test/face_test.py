# 修改后的完整流程
from deepface import DeepFace
import cv2

def safe_verify(img1_path, img2_path):
    # 强制单脸检测
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name="VGG-Face",  # 改用更稳定的模型测试[5](@ref)
        detector_backend="opencv",
        enforce_detection=True,  # 必须检测到人脸
        align=True,  # 关键对齐参数
        normalization="base",  # 确保正确归一化
        distance_metric="cosine"
    )
    return result

# 测试不同人脸
print(safe_verify("face1.png", "face2.png"))  # 应<0.4
print(safe_verify("face1.png", "face4.png"))  # 应≈0.0

# from deepface import DeepFace
# import torch
# from PIL import Image
# import numpy as np
# import os

# # 抑制TensorFlow日志
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # 初始化配置
# model_name = "Facenet"
# detector_backend = "mtcnn"

# # 加载图片（保持PIL格式）
# img1 = Image.open('face1.png').convert('RGB')
# img2 = Image.open('face2.png').convert('RGB')

# # 人脸检测与对齐（新版API）
# face_objs1 = DeepFace.extract_faces(
#     np.array(img1),
#     detector_backend=detector_backend,
#     enforce_detection=True,
#     align=True
# )

# face_objs2 = DeepFace.extract_faces(
#     np.array(img2),
#     detector_backend=detector_backend, 
#     enforce_detection=True,
#     align=True
# )

# # 获取对齐人脸（自动调整为模型输入尺寸）
# face1 = face_objs1[0]['face']  # 实际尺寸由模型决定（通常224x224）
# face2 = face_objs2[0]['face']

# # 转换为PyTorch张量
# face1_tensor = torch.tensor(face1.transpose(2,0,1)).unsqueeze(0).float()
# face2_tensor = torch.tensor(face2.transpose(2,0,1)).unsqueeze(0).float()
# print("人脸张量维度：", face1_tensor.shape)

# # 特征提取（使用Facenet模型）
# emb1 = DeepFace.represent(
#     img_path=face1,
#     model_name=model_name,
#     enforce_detection=False
# )[0]['embedding']

# emb2 = DeepFace.represent(
#     img_path=face2,
#     model_name=model_name,
#     enforce_detection=False
# )[0]['embedding']

# # 相似度计算（阈值建议参考网页3）
# emb1_tensor = torch.tensor(emb1).unsqueeze(0)
# emb2_tensor = torch.tensor(emb2).unsqueeze(0)
# cos_sim = torch.nn.functional.cosine_similarity(emb1_tensor, emb2_tensor)
# print(f"人脸相似度: {cos_sim.item():.2f}")

# # 阈值判断（Facenet建议0.40）
# threshold = 0.50
# print("验证通过" if cos_sim > threshold else "验证失败")




