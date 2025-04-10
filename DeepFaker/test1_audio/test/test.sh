echo "Test语音识别：模型只有第一次使用时需要加载"
python test_whisper.py   -f 高天.wav -m small

echo "Test声纹识别：模型只有第一次使用时需要加载"
python test_speaker.py