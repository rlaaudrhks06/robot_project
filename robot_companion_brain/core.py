# robot_companion_brain/core.py
# 로봇 반려동물 브레인 핵심 모듈: 시각, 청각, 언어 감성 분석 및 피로도 추정 기능을 통합합니다.
import cv2
import mediapipe as mp
import numpy as np
import os
import pyaudio
import threading
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from collections import deque
from .config import BrainConfig

class RobotMultimodalBrain:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        
        self.status = {
            'brightness': 120.0,
            'noise_rms': 0.0,
            'speech_detected': False,
            'last_speech_text': '대기 중...',
            'text_emotion': '-',
            'face_emotion': 'Neutral',
            'face_probability': 0.0,
            'posture_quality': 1.0,
            'fatigue_score': 0.0,       # 0 ~ 100 점
            'is_fallen': False,
            'is_immobile': False,
            'visual_weight': 0.50,
            'audio_weight': 0.30,
            'text_weight': 0.20
        }
        
        self.hip_y_history = deque(maxlen=BrainConfig.BUFFER_SIZE)
        self.jitter_history = deque(maxlen=BrainConfig.BUFFER_SIZE)
        self.fall_cooldown = 0
        self.immobile_counter = 0
        self.current_fatigue = 0.0

        self._init_ai_models()

    def _init_ai_models(self):
        print("[Brain] 반려로봇 감성 AI 컴포넌트 로딩 시작...")
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.text_emo_labels = ['공포(Fear)', '놀람(Surprise)', '분노(Angry)', '슬픔(Sad)', '중립(Neutral)', '행복(Happy)', '혐오(Disgust)']
        
        self.emotion_model = None
        try:
            from tensorflow.keras.models import load_model # type: ignore
            if os.path.exists(BrainConfig.VISION_MODEL_PATH):
                self.emotion_model = load_model(BrainConfig.VISION_MODEL_PATH)
                print(f" -> [시각] 얼굴 감정 인식 모델 탑재 완료.")
        except: pass

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BrainConfig.TEXT_MODEL_NAME)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(BrainConfig.TEXT_MODEL_NAME).to(BrainConfig.DEVICE)
            self.text_model.eval()
            print(" -> [언어] 한국어 문맥 감정 분석 모델 로드 완료.")
        except Exception as e:
            print(f" -> ❌ [언어 로드 실패]: {e}")

        try:
            self.whisper_model = whisper.load_model(BrainConfig.WHISPER_MODEL_TYPE, device=BrainConfig.DEVICE)
            print(" -> [청각] 온디바이스 Whisper STT 엔진 준비 완료.")
        except Exception as e:
            print(f" -> ❌ [청각 로드 실패]: {e}")

    def _update_dynamic_weights(self, brightness, noise_rms):
        w_visual = 0.50
        w_audio = 0.30
        w_text = 0.20
        if brightness < 70.0:
            w_visual *= (1.0 - ((70.0 - brightness) / 70.0) * 0.70)
        if noise_rms > 600.0:
            noise_penalty = min((noise_rms - 600.0) / 2000.0, 1.0)
            w_audio *= (1.0 - (noise_penalty * 0.80))
            w_text *= (1.0 - (noise_penalty * 0.40))
        total = w_visual + w_audio + w_text
        return {'visual_weight': round(w_visual/total, 2), 'audio_weight': round(w_audio/total, 2), 'text_weight': round(w_text/total, 2)}

    def _analyze_kinetics(self, pose_landmarks, h, w):
        p_q, f_d, is_fall, is_immob = 1.0, 0.0, False, False
        if not pose_landmarks: 
            # 사람이 카메라 프레임 밖으로 완전히 사라지면 피로도를 천천히 올림 (공백 누적)
            return p_q, 0.05, is_fall, is_immob
            
        lm = pose_landmarks.landmark
        l_shoulder = np.array([lm[11].x * w, lm[11].y * h])
        r_shoulder = np.array([lm[12].x * w, lm[12].y * h])
        l_hip = np.array([lm[23].x * w, lm[23].y * h])
        r_hip = np.array([lm[24].x * w, lm[24].y * h])
        nose = np.array([lm[0].x * w, lm[0].y * h])
        
        # 1. 자세 대칭성 측정 (어깨와 골반의 기울어짐)
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        hip_diff = abs(l_hip[1] - r_hip[1])
        p_q = round(max(0.0, min(1.0, 1.0 - (shoulder_diff + hip_diff) / 60.0)), 2)
        
        # [★ 감도 튜닝 1] 자세 대칭성이 무너지면(기울어지면) 피로도 증가량 증가 페널티 부여
        if p_q < 0.85:
            f_d += (0.90 - p_q) * 0.5  # 미세하게 삐딱하면 점수 상승 가속
            
        # 2. 움직임의 미세한 비틀거리거나 흔들리는 변화량(Jitter) 측정
        c_hip = (l_hip + r_hip) / 2
        self.jitter_history.append(c_hip)
        
        if len(self.jitter_history) > 1:
            d = np.linalg.norm(self.jitter_history[-1] - self.jitter_history[-2])
            
            # [★ 감도 튜닝 2] 피곤할 때 나타나는 미세 흔들림 구간(2.0 ~ 15.0 픽셀 이동)의 피로도 누적치를 3배 대폭 상승
            if 2.0 < d < 15.0:
                f_d += 0.35  # 기존 0.1에서 0.35로 대폭 강화
            elif d <= 2.0:
                # [★ 감도 튜닝 3] 완벽히 멈춰 서서 안정되었을 때 피로도가 빠지는 속도를 절반으로 감축
                f_d -= 0.02  # 기존 -0.05에서 -0.02로 완화 (피로 완화 억제)

        # 3. 낙상 및 부동 상태 체크
        self.hip_y_history.append(c_hip[1])
        if len(self.hip_y_history) >= 5:
            if (self.hip_y_history[-1] - self.hip_y_history[-5]) > (h * 0.15) and np.degrees(np.arctan2(abs((nose-c_hip)[1]), abs((nose-c_hip)[0]))) < 40.0 and self.fall_cooldown == 0:
                is_fall = True; self.fall_cooldown = 90
        if self.fall_cooldown > 0: self.fall_cooldown -= 1
        
        if len(self.jitter_history) == BrainConfig.BUFFER_SIZE:
            if np.all((np.max(np.array(self.jitter_history), axis=0) - np.min(np.array(self.jitter_history), axis=0)) < 5.0):
                self.immobile_counter += 1
                if self.immobile_counter > 150: is_immob = True
            else: self.immobile_counter = 0
            
        return p_q, f_d, is_fall, is_immob

    def analyze_frame_synchronous(self, img, results):
        if img is None: return
        h, w, _ = img.shape
        bri = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        
        p_q, f_d, is_fall, is_immob = self._analyze_kinetics(results.pose_landmarks, h, w)
        
        # 실시간 연산된 델타값을 최종 피로도 100점 만점 수치에 반영
        self.current_fatigue = max(0.0, min(100.0, self.current_fatigue + f_d))
        
        f_move_emo, f_prob = "Neutral", 100.0
        if results.face_landmarks:
            c = [(lm.x * w, lm.y * h) for lm in results.face_landmarks.landmark]
            y_min, y_max = max(0, int(min([p[1] for p in c]))-15), min(h, int(max([p[1] for p in c]))+15)
            x_min, x_max = max(0, int(min([p[0] for p in c]))-15), min(w, int(max([p[0] for p in c]))+15)
            
            if y_max > y_min and x_max > x_min:
                roi = img[y_min:y_max, x_min:x_max]
                if roi.size > 0 and self.emotion_model is not None:
                    try:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        roi_equalized = cv2.equalizeHist(roi_gray)
                        roi_resized = cv2.resize(roi_equalized, (48, 48)) / 255.0
                        preds = self.emotion_model.predict(np.reshape(roi_resized, (1, 48, 48, 1)), verbose=0)[0]
                        f_move_emo, f_prob = self.emotion_labels[np.argmax(preds)], float(preds[np.argmax(preds)] * 100)
                    except: pass
        
        with self.lock:
            self.status.update({
                'brightness': bri, 'posture_quality': p_q, 'fatigue_score': round(self.current_fatigue, 1), 
                'face_emotion': f_move_emo, 'face_probability': f_prob, 'is_immobile': is_immob
            })
            if is_fall: self.status['is_fallen'] = True
            w_dict = self._update_dynamic_weights(bri, self.status['noise_rms'])
            self.status.update(w_dict)

    def _audio_worker(self):
        p = pyaudio.PyAudio()
        try:
            device_index = p.get_default_input_device_info()['index']
            stream = p.open(format=pyaudio.paInt16, channels=BrainConfig.AUDIO_CHANNELS, rate=BrainConfig.AUDIO_RATE, input=True, input_device_index=device_index, frames_per_buffer=BrainConfig.AUDIO_CHUNK)
        except: p.terminate(); return
        noise_floor, sil_cnt, is_rec = 150.0, 0, False
        frames = []

        def _infer_async(b):
            if not self.whisper_model or not b: return
            try:
                audio_np = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size < 100: return
                txt = self.whisper_model.transcribe(audio_np, language="ko", fp16=False).get("text", "").strip()
                if txt:
                    emo = '-'
                    if self.tokenizer and self.text_model:
                        inputs = self.tokenizer([txt], return_tensors='pt', padding=True, truncation=True).to(BrainConfig.DEVICE)
                        with torch.no_grad(): emo = self.text_emo_labels[self.text_model(**inputs).logits.argmax(dim=-1).item()]
                    with self.lock: self.status['last_speech_text'] = txt; self.status['text_emotion'] = emo
            except: pass

        while self.running:
            try:
                data = stream.read(BrainConfig.AUDIO_CHUNK, exception_on_overflow=False)
                if not data: continue
                rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float64)**2))
                with self.lock: self.status['noise_rms'] = rms
                if 10 < rms < noise_floor: noise_floor = noise_floor * 0.98 + rms * 0.02
                speech = rms > (noise_floor * 2.5 + 80)
                with self.lock: self.status['speech_detected'] = speech
                if speech:
                    if not is_rec: is_rec = True; frames = []; self.status['last_speech_text'] = "듣고 있는 중..."
                    frames.append(data); sil_cnt = 0
                elif is_rec:
                    frames.append(data); sil_cnt += 1
                    if sil_cnt > 12:
                        is_rec = False
                        audio_bytes = b''.join(frames)
                        if len(audio_bytes) > 2048:
                            with self.lock: self.status['last_speech_text'] = "생각 중..."
                            threading.Thread(target=_infer_async, args=(audio_bytes,), daemon=True).start()
                        frames = []
            except: continue
        stream.close(); p.terminate()

    def start(self):
        if self.running: return
        self.running = True
        threading.Thread(target=self._audio_worker, daemon=True).start()
        print("[Package Core] 청각 루프 분리 기동 성공.")

    def stop(self): self.running = False

    def get_companion_interpretation(self):
        with self.lock: st = self.status.copy()
        
        energy_score = max(0.0, min(100.0, 100.0 - st['fatigue_score']))
        face_emo = st['face_emotion']
        text_emo = st['text_emotion']
        v_w, t_w = st['visual_weight'], st['text_weight']
        final_emotion = face_emo if v_w >= 0.25 or text_emo == '-' else text_emo.split('(')[0]
        
        user_mood, recommended_interaction = "NEUTRAL", "STAY_QUIET"
        
        if st['is_fallen']: 
            user_mood, recommended_interaction = "SURPRISED_OR_FALLEN", "EMERGENCY_APPROACH"
        # [★ 융합 임계값 조정] 피로도가 민감해진 만큼, 피로 누적 가이드 발동선도 여유롭게 최적화
        elif energy_score < 50.0:  # 기존 40.0% 미만에서 50.0% 미만으로 기준 확대
            if face_emo in ['Sad', 'Angry', 'Fear'] or '슬픔' in text_emo: 
                user_mood, recommended_interaction = "EXHAUSTED_SAD", "CONSOLATION_APPROACH"
            else: 
                user_mood, recommended_interaction = "TIRED", "STAY_QUIET"
        elif face_emo == 'Happy' or '행복' in text_emo: 
            user_mood, recommended_interaction = "JOYFUL", "CHEER_UP"
        elif face_emo == 'Surprise': 
            user_mood, recommended_interaction = "STARTLED", "TILTING_HEAD"
        elif not st['speech_detected'] and face_emo == 'Neutral': 
            user_mood, recommended_interaction = "LONELY", "GENTLE_TALK"
            
        return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "robot_action_target": {"user_mood": user_mood, "dominant_emotion": final_emotion, "recommended_interaction": recommended_interaction, "last_speech_text": st['last_speech_text']}, "sensor_trust_matrix": {"visual_weight": v_w, "audio_weight": st['audio_weight'], "text_weight": t_w}, "internal_analytics": {"emotional_energy": round(energy_score, 1), "physical_fatigue": st['fatigue_score'], "is_dark_now": True if st['brightness'] < 40.0 else False, "is_noisy_now": True if st['noise_rms'] > 600.0 else False}}