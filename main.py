# main.py
# 로봇 반려동물 브레인 모니터링 및 시각화 메인 스크립트입니다.
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image
from robot_companion_brain import RobotMultimodalBrain

def put_korean_text(img, text, position, font_size, color):
    b, g, r = color
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    try: font = ImageFont.truetype(font_path, font_size)
    except IOError: font = ImageFont.load_default()
    if img is None or img.size == 0: return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img_pil).text(position, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    brain = RobotMultimodalBrain()
    brain.start()
    
    print("\n[Robot Main] 단일 스레드 안전화 모드로 모니터링 창을 실행합니다...")
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            h, w, _ = frame.shape

            # 1. 메인 루프에서 미디어파이프 연산 선행 수행
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # 2. 수행 결과를 브레인의 감정 분석 함수로 실시간 동기 전달 (안전)
            brain.analyze_frame_synchronous(frame, results)
            
            # 3. 데이터 가져오기
            status = brain.get_companion_interpretation()
            action_guide = status['robot_action_target']
            trust_matrix = status['sensor_trust_matrix']
            analytics = status['internal_analytics']

            # 화면 뼈대 그리기
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                c = [(lm.x * w, lm.y * h) for lm in results.face_landmarks.landmark]
                cv2.rectangle(frame, (max(0, int(min([p[0] for p in c]))-15), max(0, int(min([p[1] for p in c]))-15)), (min(w, int(max([p[0] for p in c]))+15), min(h, int(max([p[1] for p in c]))+15)), (0, 255, 0), 1)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 대시보드 박스 렌더링
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 230), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"USER MOOD: {action_guide['user_mood']}", (20, 35), font, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, f"EMOTION (FUSION): {action_guide['dominant_emotion']}", (20, 60), font, 0.5, (100, 255, 100), 1)
            cv2.putText(frame, f"TRUST -> V:{trust_matrix['visual_weight']:.2f} | A:{trust_matrix['audio_weight']:.2f} | T:{trust_matrix['text_weight']:.2f}", (20, 90), font, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Energy: {analytics['emotional_energy']}% | Fatigue: {analytics['physical_fatigue']}", (20, 115), font, 0.45, (150, 200, 255), 1)
            
            frame = put_korean_text(frame, f"STT 내용: {action_guide.get('last_speech_text', '대기 중...')}", (20, 140), 13, (255, 150, 255))
            frame = put_korean_text(frame, f"▶ ROBOT ACTION GUIDE:", (20, 170), 12, (255, 255, 255))
            frame = put_korean_text(frame, f"  \"{action_guide['recommended_interaction']}\"", (20, 195), 14, (0, 255, 255))

            cv2.imshow('Robot Companion Brain - Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    brain.stop()
    print("\n[Robot Main] 안전 종료 완료.")

if __name__ == "__main__":
    main()