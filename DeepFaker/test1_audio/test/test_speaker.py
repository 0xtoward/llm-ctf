# speaker_verify.py
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier

# 1. 初始化预训练模型（ECAPA-TDNN）
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models"
)

# 2. 提取声纹特征
def extract_embedding(audio_path):
    # 加载音频（保持原始声道）
    signal, fs = torchaudio.load(audio_path)
    
    # 声道转换（网页[3]的优化方案）
    if signal.shape[0] > 1:  # 多声道情况
        signal = torch.mean(signal, dim=0, keepdim=True)  # 合并为单声道
    
    # 采样率转换（网页[5]的参数规范）
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=fs,
            new_freq=16000,
            resampling_method="sinc_interpolation"  # 网页[5]推荐方法
        )
        signal = resampler(signal)
    
    # 提取特征（需确保输入为单声道）
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze(0)


# 3. 计算相似度
def verify_speaker(audio1, audio2, threshold=0.85):
    emb1 = extract_embedding(audio1)
    emb2 = extract_embedding(audio2)
    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
    # 返回相似度得分和判定结果
    return similarity.item(), similarity.item() > threshold

if __name__ == "__main__":
    # 输入两段音频路径
    audio_path1 = "高天.wav"
    audio_path2 = "高天1.mp3"
    
    # 执行验证
    score, is_same_speaker = verify_speaker(audio_path1, audio_path2)
    if is_same_speaker:
        print(f"✅ 两段语音来自同一人，相似度得分：{score:.2f}")
    else:
        print(f"❌ 语音来自不同人，相似度得分：{score:.2f}")


    audio_path3 = "仙翁.mp3"
    
    # 执行验证
    score, is_same_speaker = verify_speaker(audio_path1, audio_path3)
    if is_same_speaker:
        print(f"✅ 两段语音来自同一人，相似度得分：{score:.2f}")
    else:
        print(f"❌ 语音来自不同人，相似度得分：{score:.2f}")
