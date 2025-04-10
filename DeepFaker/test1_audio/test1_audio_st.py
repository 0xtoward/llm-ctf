import streamlit as st
import os
import hashlib
import torch
import torchaudio
from pydub import AudioSegment
import io
import whisper
from speechbrain.inference import EncoderClassifier
from difflib import SequenceMatcher



# 获取当前脚本的绝对路径目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本目录: {SCRIPT_DIR}")
# 修改后的路径配置
BASE_AUDIO_PATH = os.path.join(SCRIPT_DIR, "video", "仙翁纯享.mp3")
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp_uploads")
VIDEO_PATH = os.path.join(SCRIPT_DIR, "video", "nezha2.mp4")
os.makedirs(TEMP_DIR, exist_ok=True)


# 安全配置
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# 初始化模型（带缓存）
@st.cache_resource
def load_models():
    """加载语音模型并返回"""
    # 声纹识别模型
    spk_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models"
    )
    
    # 语音识别模型
    asr_model = whisper.load_model("small")
    
    # 预加载基准声纹（维度修复关键）
    base_signal, fs = torchaudio.load(BASE_AUDIO_PATH)
    if base_signal.shape[0] > 1:
        base_signal = torch.mean(base_signal, dim=0, keepdim=True)
    if fs != 16000:
        base_signal = torchaudio.functional.resample(base_signal, fs, 16000)
    base_emb = spk_classifier.encode_batch(base_signal)
    base_emb = base_emb.squeeze(0).flatten()  # 维度调整 [1, 192] -> [192]
    
    return spk_classifier, asr_model, base_emb

spk_model, asr_model, BASE_EMBEDDING = load_models()

def audio_preprocessing(uploaded_file):
    """优化音频预处理流程"""
    try:
        audio = AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()))
        
        # 增强预处理：去除静音段
        processed = (audio.set_frame_rate(16000)
                    .set_channels(1)
                    .normalize())
        
        # 保存预处理文件
        converted_path = os.path.join(TEMP_DIR, f"conv_{hashlib.md5(uploaded_file.getvalue()).hexdigest()}.wav")
        processed.export(converted_path, format="wav")
        return converted_path
    except Exception as e:
        raise RuntimeError(f"音频预处理失败: {str(e)}")

def extract_voiceprint(audio_path):
    """维度修复后的声纹特征提取"""
    try:
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        
        # 维度处理流程优化
        embeddings = spk_model.encode_batch(signal)
        return embeddings.squeeze(0).flatten()  # 输出形状 [192]
    except Exception as e:
        raise RuntimeError(f"声纹提取失败: {str(e)}")

def verify_speaker(user_emb):
    """修复后的相似度计算"""
    try:
        # 确保维度匹配
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)
        
        similarity = torch.nn.functional.cosine_similarity(
            BASE_EMBEDDING.unsqueeze(0), 
            user_emb, 
            dim=1  # 修正维度参数
        )
        return max(similarity.item(), 0.0)
    except Exception as e:
        raise RuntimeError(f"声纹验证失败: {str(e)}")

def transcribe_audio(audio_path):
    """增强版语音转文字"""
    try:
        result = asr_model.transcribe(
            audio_path,
            language="zh",
            initial_prompt="以下是普通话句子",
            fp16=False
        )
        return result["text"].strip()
    except Exception as e:
        raise RuntimeError(f"语音识别失败: {str(e)}")

# 网页界面
st.title("八卦宫语音验证 v1.0")
TARGET_TEXT = "我乃无量仙翁 师弟别来无恙"  # 固定验证文本

video_file = open(VIDEO_PATH, 'rb')
video_bytes = video_file.read()
st.video(video_bytes, format="video/mp4") 

# 素材下载区
st.markdown("### 🎧 素材下载")
col1, col2 = st.columns(2)

with col1:
    with open(VIDEO_PATH, "rb") as f:
        st.download_button(
            label="下载哪吒素材",
            data=f,
            file_name="nezha2.mp4",
            mime="video/mp4",
            help="包含声纹特征的参考视频"
        )

with col2:
    with open(BASE_AUDIO_PATH, "rb") as f:
        st.download_button(
            label="下载基准音频",
            data=f,
            file_name="仙翁纯享.mp3",
            mime="audio/mpeg",
            help="目标声纹的原始录音"
        )

# 验证目标声明
st.markdown(f"""
<div style="background:#f0f2f6;padding:20px;border-radius:10px">
    <h4>🎯 验证目标要求</h4>
    <p>1. 使用<b>仙翁声纹特征</b>录制语音（相似度≥阈值%）</p>
    <p>2. 清晰读出以下文本：<code>{TARGET_TEXT}</code>（匹配度≥70%）</p>
</div>
""", unsafe_allow_html=True)

# 验证表单
with st.form("verify_form"):
    uploaded_file = st.file_uploader(
        "上传验证语音（MP3/WAV）",
        type=["mp3", "wav"],
        help="文件需同时满足声纹和文本要求"
    )
    
    if st.form_submit_button("开始验证"):
        if not uploaded_file:
            st.warning("请先上传语音文件")
            st.stop()
            
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("文件大小超过10MB限制")
            st.stop()

        try:
            # 预处理阶段
            conv_path = audio_preprocessing(uploaded_file)
            
            # 音频长度验证
            audio_length = AudioSegment.from_file(conv_path).duration_seconds
            if audio_length < 2.0:
                st.error("❌ 语音时长不足2秒，请重新录制")
                st.stop()
            
            # 并行处理双验证
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("🔍 声纹分析中..."):
                    user_emb = extract_voiceprint(conv_path)
                    sim_score = verify_speaker(user_emb)
                    
            with col2:
                with st.spinner("🔤 文本识别中..."):
                    transcript = transcribe_audio(conv_path)
                    text_match = SequenceMatcher(
                        None, 
                        TARGET_TEXT, 
                        transcript
                    ).ratio()
            
            # 显示验证结果
            st.subheader("验证结果")
            rs_col1, rs_col2 = st.columns(2)
            
            with rs_col1:
                st.metric("声纹相似度", 
                         f"{sim_score+(1-sim_score)*0.5:.2%}", #正则化声纹相似度
                         delta="≥75%" if sim_score >= 0.5 else None)
                
            with rs_col2:
                st.metric("文本匹配度",
                         f"{text_match:.2%}",
                         delta="≥70%" if text_match >= 0.7 else None)
            
            st.divider()
            st.subheader("识别内容")
            st.code(transcript)
            st.caption(f"目标文本: {TARGET_TEXT}")
            
            # 最终判定
            if sim_score >= 0.5 and text_match >= 0.7:
                flag = hashlib.sha256(conv_path.encode()).hexdigest()[:16]
                st.success(f"✅ 验证通过! FLAG: CTF{{VoiceAuth_{flag}}}")
                st.balloons()
            else:
                st.error("❌ 验证失败：未通过双重校验")
                
        except Exception as e:
            st.error(f"系统错误: {str(e)}")
        finally:
            if os.path.exists(conv_path):
                os.remove(conv_path)