# robot_companion_brain/config.py
# 화면 해상도, 오디오 설정, 모델 경로 등 브레인 관련 설정을 정의하는 모듈입니다.
import os
import torch

class BrainConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # [안전 패치] 맥북 M시리즈 세그폴트 방지를 위해 640x480 안전 해상도로 고정
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    BUFFER_SIZE = 30  

    AUDIO_CHANNELS = 1
    AUDIO_RATE = 16000
    AUDIO_CHUNK = 1024

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VISION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.h5") 
    TEXT_MODEL_NAME = 'dlckdfuf141/korean-emotion-kluebert-v2'
    WHISPER_MODEL_TYPE = "base"