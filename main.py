import cv2
import numpy as np
import time
from database import DatabaseManager
from health_engine import HealthEngine
from emotion_engine import EmotionEngine
from face_engine import FaceEngine

def draw_hud(frame, bpm, pulse_signal, emotion, emotion_conf, ear, mar, pitch, yaw, roll, focus, fatigue, stress, session_id):
    """Draws a clean, futuristic Heads-Up Display (HUD) overlay on the video feed."""
    h, w, _ = frame.shape
    
    # 1. Top Bar: Session & Status
    cv2.rectangle(frame, (15, 15), (w - 15, 60), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 15), (w - 15, 60), (100, 100, 100), 1)
    
    # Recording Badge
    cv2.circle(frame, (35, 37), 7, (0, 0, 255), -1)
    cv2.putText(frame, f"REC [Session #{session_id}]", (50, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(frame, f"FOCUS: {focus:.0f}%", (w - 180, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150) if focus > 65 else (0, 140, 255), 2)

    # 2. Left Panel: Health & Vitals
    cv2.rectangle(frame, (15, 75), (280, 260), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 75), (280, 260), (100, 100, 100), 1)
    cv2.putText(frame, "PHYSIOLOGY & VITALS", (25, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    bpm_text = f"{bpm:.1f} BPM" if bpm else "Calibrating..."
    cv2.putText(frame, f"Heart Rate: {bpm_text}", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if bpm else (180, 180, 180), 2)
    cv2.putText(frame, f"Stress State: {stress}", (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    
    fatigue_color = (0, 255, 0) if "Normal" in fatigue else ((0, 0, 255) if "DROWSY" in fatigue else (0, 165, 255))
    cv2.putText(frame, f"Alertness: {fatigue}", (25, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fatigue_color, 2)
    cv2.putText(frame, f"EAR (Blink): {ear:.2f} | MAR: {mar:.2f}", (25, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, f"Pose: P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}", (25, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # 3. Right Panel: Emotion & Affective State
    cv2.rectangle(frame, (w - 295, 75), (w - 15, 200), (20, 20, 20), -1)
    cv2.rectangle(frame, (w - 295, 75), (w - 15, 200), (100, 100, 100), 1)
    cv2.putText(frame, "AFFECTIVE / EMOTION", (w - 285, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(frame, f"State: {emotion}", (w - 285, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    cv2.putText(frame, f"Confidence: {emotion_conf*100:.0f}%", (w - 285, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 4. Bottom Waveform Display for Pulse
    if len(pulse_signal) > 10:
        wave_box_x1, wave_box_y1 = 15, h - 85
        wave_box_x2, wave_box_y2 = w - 15, h - 15
        cv2.rectangle(frame, (wave_box_x1, wave_box_y1), (wave_box_x2, wave_box_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (wave_box_x1, wave_box_y1), (wave_box_x2, wave_box_y2), (60, 60, 60), 1)
        cv2.putText(frame, "rPPG Pulse Waveform", (wave_box_x1 + 10, wave_box_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        sig_slice = pulse_signal[-120:]  # Last 120 samples
        pts = []
        box_w = wave_box_x2 - wave_box_x1 - 20
        box_h = wave_box_y2 - wave_box_y1 - 25
        base_y = wave_box_y2 - 12
        
        for i, val in enumerate(sig_slice):
            px = int(wave_box_x1 + 10 + (i / max(1, len(sig_slice) - 1)) * box_w)
            py = int(base_y - (val + 2.0) / 4.0 * box_h)
            py = max(wave_box_y1 + 20, min(wave_box_y2 - 5, py))
            pts.append((px, py))
            
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], (0, 255, 100), 2)

    # Keybinds footer
    cv2.putText(frame, "Press 'Q' to Quit & Save Log | 'E' to Export CSV", (w // 2 - 180, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def main():
    db = DatabaseManager()
    session_id = db.start_session(notes="Live webcam session")
    print(f"[+] Started Tracking Session #{session_id}. Database: tracker_history.db")
    
    health = HealthEngine(fps=30)
    emotion_eng = EmotionEngine()
    face_eng = FaceEngine()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Error: Could not access webcam. Please check device permissions.")
        return

    # Try setting optimal webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally (left-right mirror)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = face_eng.process_frame(frame)
            
            bpm = None
            filtered_signal = []
            emotion = "No Face"
            emotion_conf = 0.0
            ear_avg = 0.0
            mar = 0.0
            pitch, yaw, roll, focus = 0.0, 0.0, 0.0, 0.0
            fatigue = "No Face"
            stress = "Unknown"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 1. 3D Head Pose & Focus
                pitch, yaw, roll, focus = face_eng.estimate_head_pose(landmarks, w, h)
                
                # 2. Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)
                ear_l = health.calculate_ear(landmarks, health.LEFT_EYE_IDX, w, h)
                ear_r = health.calculate_ear(landmarks, health.RIGHT_EYE_IDX, w, h)
                mar = health.calculate_mar(landmarks, w, h)
                
                # 3. Health & Fatigue
                ear_avg, mar, fatigue, stress, is_drowsy = health.evaluate_fatigue_and_stress(ear_l, ear_r, mar, health.current_bpm)
                if is_drowsy:
                    db.log_event("Drowsiness Alert", f"EAR dropped below threshold for extended frames: {ear_avg:.2f}")

                # 4. rPPG Forehead extraction
                raw_color = health.extract_forehead_roi(frame, landmarks, w, h)
                bpm, filtered_signal = health.update_rppg(raw_color)

                # 5. Emotion Classification
                emotion, emotion_conf, valence = emotion_eng.analyze_emotion(frame, landmarks, ear_avg, mar, pitch, w, h)

                # 6. Database Logging (Every second automatically)
                db.log_metric(bpm, emotion, emotion_conf, ear_avg, mar, pitch, yaw, roll, focus, fatigue, stress)

            # Draw HUD
            draw_hud(frame, bpm, filtered_signal, emotion, emotion_conf, ear_avg, mar, pitch, yaw, roll, focus, fatigue, stress, session_id)
            cv2.imshow("AI Real-Time Emotion, Health & Face Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                csv_file = db.export_session_to_csv(session_id)
                print(f"[+] Current session exported to: {csv_file}")

    finally:
        saved_id = db.end_session()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[+] Session #{saved_id} successfully finalized and saved in database.")

if __name__ == "__main__":
    main()
