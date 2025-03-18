# test_whisper.py
import whisper
import argparse
import os
import time

def transcribe_audio(audio_path, model_type="base"):
    """
    语音转文字核心功能（基于网页2的官方实现优化）
    参数：
        audio_path: 音频文件路径（支持mp3/wav等格式）
        model_type: whisper模型类型，可选值：tiny/base/small/medium/large
    """
    try:
        # 模型加载（网页2的加载方式改进）
        start_time = time.time()
        print(f"⏳ 正在加载{model_type}模型...")
        model = whisper.load_model(model_type)
        print(f"✅ 模型加载完成，耗时：{time.time()-start_time:.1f}秒")

        # 执行语音识别（网页2的转写参数优化）
        print(f"🔊 开始识别文件：{audio_path}")
        result = model.transcribe(audio_path, 
                                fp16=False,  # 兼容CPU模式
                                language="zh",  # 指定中文识别
                                initial_prompt="以下是普通话的句子" # 提示语优化
                                )
        
        # 返回识别结果
        return result["text"]

    except FileNotFoundError:
        raise Exception(f"错误：音频文件不存在 {audio_path}")
    except Exception as e:
        raise Exception(f"识别失败：{str(e)}")

if __name__ == "__main__":
    # 命令行参数解析（新增功能）
    parser = argparse.ArgumentParser(description="Whisper语音识别测试脚本")
    parser.add_argument("-f", "--file", required=True, help="音频文件路径")
    parser.add_argument("-m", "--model", default="base", 
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="选择模型版本（默认base）")
    args = parser.parse_args()

    # 文件存在性验证（新增安全检查）
    if not os.path.exists(args.file):
        print(f"❌ 错误：文件不存在 {args.file}")
        exit(1)

    # 执行识别
    try:
        text = transcribe_audio(args.file, args.model)
        print("\n📝 识别结果：")
        print("-" * 40)
        print(text)
        print("-" * 40)
    except Exception as e:
        print(f"❌ 发生错误：{str(e)}")