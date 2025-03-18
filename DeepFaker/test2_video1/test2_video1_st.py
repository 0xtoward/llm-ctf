import streamlit as st
import os
import hashlib
import torch
import torchaudio
import cv2
import numpy as np
import subprocess
import whisper
from deepface import DeepFace
from difflib import SequenceMatcher
from speechbrain.inference import EncoderClassifier

# 安全配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
TEMP_DIR = "./temp_uploads"
BASE_VIDEO_PATH = "./video/高天.mp4"
TARGET_TEXT = "没有网络安全就没有国家安全"
os.makedirs(TEMP_DIR, exist_ok=True)

# 初始化模型（带缓存）
@st.cache_resource
def load_models():
    """加载多模态验证模型"""
    # 声纹识别模型
    spk_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models"
    )
    
    # 语音识别模型
    asr_model = whisper.load_model("small")
    
    # 预加载目标特征
    target_face = extract_target_features(BASE_VIDEO_PATH)
    target_voice = extract_voice_features(BASE_VIDEO_PATH)
    
    return {
        "spk": spk_classifier,
        "asr": asr_model,
        "target_face": target_face,
        "target_voice": target_voice
    }

def extract_target_features(video_path):
    """使用DeepFace提取目标视频特征"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 2)  # 每秒采样2帧
    
    embeddings = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            try:
                # 使用MTCNN检测器
                embedding_obj = DeepFace.represent(
                    img_path=frame,
                    model_name="VGG-Face",
                    detector_backend="mtcnn",
                    enforce_detection=True,
                    align=True
                )
                if embedding_obj:
                    embeddings.append(embedding_obj[0]['embedding'])
            except Exception as e:
                print(f"目标视频特征提取失败: {str(e)}")
    
    cap.release()
    return np.mean(embeddings, axis=0) if embeddings else None

def extract_voice_features(video_path):
    """提取目标声纹特征"""
    audio_path = os.path.join(TEMP_DIR, "target_audio.wav")
    
    # 使用FFmpeg提取音频
    cmd = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    subprocess.call(cmd, shell=True)
    
    signal, fs = torchaudio.load(audio_path)
    signal = torchaudio.functional.resample(signal, fs, 16000)
    return models["spk"].encode_batch(signal).squeeze(0).flatten()

def process_uploaded_video(uploaded_file):
    """处理上传的视频文件"""
    # 保存临时文件
    video_path = os.path.join(TEMP_DIR, f"upload_{hashlib.md5(uploaded_file.getvalue()).hexdigest()}.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # 提取视频帧（每秒2帧）
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 2)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 提取音频
    audio_path = os.path.join(TEMP_DIR, f"audio_{os.path.basename(video_path)}.wav")
    cmd = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    subprocess.call(cmd, shell=True)
    
    return video_path, frames, audio_path

def verify_face(frames):
    """人脸特征验证（DeepFace实现）"""
    user_embeddings = []
    
    for frame in frames:
        try:
            # 转换颜色空间并提取特征
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding_obj = DeepFace.represent(
                img_path=rgb_frame,
                model_name="VGG-Face",
                detector_backend="mtcnn",
                enforce_detection=True,
                align=True
            )
            if embedding_obj:
                user_embeddings.append(embedding_obj[0]['embedding'])
        except Exception as e:
            print(f"用户视频帧处理失败: {str(e)}")
    
    if not user_embeddings or models["target_face"] is None:
        return 0.0
    
    # 计算平均相似度
    user_avg = np.mean(user_embeddings, axis=0)
    similarity = np.dot(user_avg, models["target_face"]) / (
        np.linalg.norm(user_avg) * np.linalg.norm(models["target_face"]))
    
    return float(similarity)

def verify_voice(audio_path):
    """声纹特征验证"""
    signal, fs = torchaudio.load(audio_path)
    signal = torchaudio.functional.resample(signal, fs, 16000)
    user_emb = models["spk"].encode_batch(signal).squeeze(0).flatten()
    return torch.nn.functional.cosine_similarity(
        models["target_voice"].unsqueeze(0),
        user_emb.unsqueeze(0)
    ).item()

def verify_text(audio_path):
    """语音内容验证（仅字符相似度）"""
    result = models["asr"].transcribe(
        audio_path,
        language="zh",
        initial_prompt="网络安全相关语句"
    )
    transcript = result["text"].strip()
    return SequenceMatcher(None, TARGET_TEXT, transcript).ratio()

# 初始化模型
models = load_models()

# 网页界面
st.title("昆仑镜-多模态验证系统 v3.0")
st.video(BASE_VIDEO_PATH, format="video/mp4", start_time=0)

# 验证表单
with st.form("verify_form"):
    uploaded_file = st.file_uploader(
        "上传验证视频（MP4格式）",
        type=["mp4"],
        help="需同时满足人脸、声纹、语音内容要求"
    )
    
    if st.form_submit_button("开始验证"):
        if not uploaded_file:
            st.warning("请上传验证视频")
            st.stop()
            
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("文件大小超过50MB限制")
            st.stop()

        try:
            # 处理上传视频
            video_path, frames, audio_path = process_uploaded_video(uploaded_file)
            
            # 并行验证
            with st.spinner("🔍 综合验证中..."):
                face_score = max(verify_face(frames), 0)
                voice_score = max(verify_voice(audio_path), 0)
                text_score = max(verify_text(audio_path), 0)
            
            # 显示结果
            col1, col2, col3 = st.columns(3)
            col1.metric("人脸相似度", f"{face_score:.2%}", delta="≥90%")
            col2.metric("声纹匹配度", f"{voice_score:.2%}", delta="≥85%")
            col3.metric("文本匹配度", f"{text_score:.2%}", delta="≥70%")
            
            # 最终判定
            if all([
                face_score >= 0.9,
                voice_score >= 0.85,
                text_score >= 0.7
            ]):
                flag = hashlib.sha256(video_path.encode()).hexdigest()[:16]
                st.success(f"✅ 验证成功! FLAG: CTF{{TriAuth_{flag}}}")
                st.balloons()
            else:
                fail_reasons = []
                if face_score < 0.9: fail_reasons.append("人脸不匹配")
                if voice_score < 0.85: fail_reasons.append("声纹不符") 
                if text_score < 0.7: fail_reasons.append("内容错误")
                st.error(f"❌ 验证失败：{' + '.join(fail_reasons)}")
                
        except Exception as e:
            st.error(f"系统错误: {str(e)}")
        finally:
            # 清理临时文件
            for path in [video_path, audio_path]:
                if path and os.path.exists(path):
                    os.remove(path)