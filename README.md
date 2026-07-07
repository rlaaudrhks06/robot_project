# 🤖 반려로봇 감성 AI 패키지 브레인 (Robot Companion Brain)

Apple Silicon (M시리즈) 맥북 및 임베디드 가속 환경에 최적화된 반려로봇용 실시간 멀티모달(시각+청각+문맥) 감성 지능 처리 엔진입니다. 카메라를 통한 표정 및 뼈대 인식, 마이크를 통한 온디바이스 음성인식(STT) 결과를 규칙 기반 감정 융합 매트릭스로 연산하여 로봇의 최종 행동 지침을 도출합니다.

---

## 🌟 핵심 아키텍처 및 특징

*   **크래시 프리 동기식 파이프라인 (Single-Thread Safety)**: 백엔드 OpenCV 및 MediaPipe C++ 컨텍스트의 메모리 경합을 원천 차단하기 위해, 메인 루프에서 영상 캡처와 프레임 처리를 제어하고 AI 인지 엔진으로 동기 전달하는 아키텍처를 채택했습니다.
*   **2분법 감정 융합 매트릭스**: 복잡한 인공지능 확률값의 불확실성을 제어하기 위해, 발화 맥락(긍정/부정)과 얼굴 표정(행복/부정)을 이진화 맵으로 압축한 뒤 실시간 에너지(Energy) 및 지침 수치(Fatigue)에 누적 반영합니다.
*   **실시간 움직임 피로도 분석 (Kinetics)**: 미디어파이프 포즈 랜드마크의 실시간 변화량(Jitter)과 상체 대칭성(Posture Quality)을 추적하여 거북목, 어깨 처짐, 무기력한 흔들림 발생 시 피로도 점수를 가속 적립합니다.
*   **온디바이스 가볍고 빠른 STT/NLP**: OpenAI Whisper 엔진과 한국어 문맥 감정 KLUE-BERT 모델을 연동하여, 말소리가 끝난 후 약 0.5초 침묵 시 비동기 스레드로 떼어 빠르게 텍스트 감정을 파악합니다.

---

## 📂 프로젝트 디렉토리 구조

```
robot_project/
├── ai_env/                       # Python 3.9 가상환경 (M2 최적화)
├── main.py                       # 메인 GUI 실행 스크립트 (카메라 & HUD 화면 표시)
└── robot_companion_brain/        # 멀티모달 브레인 패키지 코어
    ├── __init__.py               # 패키지 초기화 노출부
    ├── config.py                 # 하드웨어, 해상도, 모델 경로 설정 파일
    ├── core.py                   # 시각/청각/언어 데이터 동기식 융합 처리 연산 엔진
    └── models/
        └── emotion_model.h5      # 2D 얼굴 감정 인식 Keras 웨이트 파일
```
실행 및 모니터링 방법
가상환경 (ai_env) 상태에서 프로젝트 루트 경로로 이동한 뒤 메인 런쳐를 실행합니다. -> python main.py

실시간 대시보드 HUD 가이드
프로그램이 가동되면 상단 좌측에 실시간 AI 매트릭스가 반투명 레이어로 오버레이됩니다.

```
USER MOOD: 융합 엔진이 최종 판단한 주인의 감태 (NEUTRAL, JOYFUL, TIRED, LONELY, SURPRISED_OR_FALLEN)

EMOTION (FUSION): 현재 우선순위 필터링을 통과한 도미넌트 표정/언어 감정

TRUST: 주변 환경(조도 및 마이크 소음)에 따라 동적으로 가중치가 변하는 센서 신뢰 지수 (V: 시각, A: 오디오, T: 텍스트)

Energy / Fatigue: 실시간 표정 및 마이크 발화 강도(긍정/부정)에 따라 실시간 충전/방전되는 배터리 스타일 스코어

▶ ROBOT ACTION GUIDE: 로봇 바디 하드웨어 모터 및 스피커로 전달할 실시간 주행 및 태스크 행동 명령어

시스템 종료: 카메라 렌더링 화면 창을 마우스로 클릭한 뒤, 키보드 단축키 q를 누르면 안전하게 백그라운드 리소스를 릴리스하고 종료됩니다.
```


규칙 기반 감정 반영 메커니즘 수식

1. 시각 표정 프레임 연산 (실시간 누적)
```
Face == 'Happy' -> ΔEnergy = +0.3, ΔFatigue = -0.4

Face in ['Angry', 'Sad', 'Fear', ...] -> ΔEnergy = -0.3, ΔFatigue = +0.3
```
2. 청각 언어 문맥 연산 (비동기 이벤트 트리거)
```
Text_Emotion in [행복, 놀람, 중립] -> Energy += 15.0, Fatigue -= 10.0

Text_Emotion in [공포, 분노, 슬픔, 혐오] -> Energy -= 15.0, Fatigue += 20.0
```

# 반려로봇 감성 AI 패키지 설치 가이드

이 가이드는 프로젝트 구동에 필요한 Python 가상환경을 구축하고 관련 AI 패키지들을 안정적으로 설치하는 방법을 안내합니다.

---

## 1. 가상환경 생성 및 활성화

프로젝트 폴더 내에 독립된 가상환경을 생성하고 활성화합니다.

```
# 프로젝트 폴더로 이동
cd ~/Desktop/robot_project

# 가상환경 생성 (ai_env)
python3 -m venv ai_env

# 가상환경 활성화
source ai_env/bin/activate

# 1. 기본 빌드 도구 업데이트
pip install --upgrade pip setuptools wheel

# 2. 텐서플로우 및 프로토콜 버퍼 설치
pip install tensorflow==2.15.0 protobuf==4.25.3

# 3. 미디어파이프 및 OpenCV 설치
pip install mediapipe==0.10.14 opencv-python==4.10.0.84

# 4. 음성인식(Whisper) 및 AI 모델 관련 패키지 설치
pip install openai-whisper transformers torch torchvision pyaudio
