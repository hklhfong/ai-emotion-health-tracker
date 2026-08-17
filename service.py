import cv2
import numpy as np
import time
import os
import sys
import threading
from database import DatabaseManager
from health_engine import HealthEngine
from emotion_engine import EmotionEngine
from face_engine import FaceEngine
from report_organizer import ReportOrganizer
from notifier import NotificationManager

class BackgroundTrackerService:
    """
    Ultra-Low Resource AI Tracker Daemon:
    - Dual Operation Modes:
        1. Low-Power Background Mode (Default): Runs inference at ~1.0 Hz (1 sample per second)
           reducing CPU utilization to < 2-4% and saving battery.
        2. Live HUD Mode (--gui): Runs smooth visual rendering for interactive testing.
    - Deep Learning Emotion Tracking (AffectNet ONNX)
    - 3D Head Pose + Iris Gaze Attention Fusion
    - Contactless rPPG Vitals & Multimodal Stress Index (MSI: 0 - 100)
    - Auto 15-Minute Standby & Session Organizer
    """
    def __init__(self, headless=True, no_face_timeout_sec=900, retry_interval=3.0, sample_interval_sec=1.0):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        self.headless = headless
        self.no_face_timeout_sec = no_face_timeout_sec
        self.retry_interval = retry_interval
        self.sample_interval_sec = sample_interval_sec  # 1.0 second cadence for low resource usage
        self.running = True
        
        self.db = DatabaseManager()
        self.health = HealthEngine(fps=30)
        self.emotion_eng = EmotionEngine()
        self.face_eng = FaceEngine()
        self.organizer = ReportOrganizer()
        self.notifier = NotificationManager(cooldown_seconds=180)
        
        self.cap = None
        self.session_id = None
        self.camera_busy = False
        
        self.last_face_seen_time = time.time()
        self.in_standby = False
        
        self.high_stress_count = 0
        self.high_stress_threshold = 3  # 3 consecutive high stress intervals (~3 seconds)

    def open_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            # Lower resolution in background mode to save processing power
            if self.headless:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            else:
                cap.release()
        return None

    def start(self):
        print("[+] AI Tracker Service initialized.")
        print(f"[+] Low-Resource Cadence: {self.sample_interval_sec}s interval in background (CPU ~2%)")
        print(f"[+] 15-Minute Standby & Report Organizer: Active")
        print(f"[+] Multimodal Stress Index & Eye Gaze: Active")

        self.session_id = self.db.start_session(notes="Low-resource background tracking session")
        self.last_face_seen_time = time.time()
        consecutive_read_failures = 0
        last_process_time = 0

        while self.running:
            # --- STANDBY CHECK (Triggered after 15 mins of no face) ---
            if self.in_standby:
                from input_monitor import get_idle_duration_seconds
                idle_sec = get_idle_duration_seconds()
                
                # If user is still completely away from PC (no mouse/keyboard input), sleep deeply (0% CPU)
                if idle_sec > 5.0:
                    time.sleep(1.0)
                    continue

                # User moved mouse or typed! Probe camera to verify face presence and wake up
                print("\n[+] Mouse / keyboard activity detected! Checking camera for user presence...")
                probe_cap = self.open_camera()
                if probe_cap:
                    ret, probe_frame = probe_cap.read()
                    if ret and probe_frame is not None:
                        probe_frame = cv2.flip(probe_frame, 1)
                        res = self.face_eng.process_frame(probe_frame)
                        if res.multi_face_landmarks:
                            print("[+] Face confirmed! Waking up from standby and starting new session.")
                            self.in_standby = False
                            self.session_id = self.db.start_session(notes="Auto-resumed after mouse activity & face detected")
                            self.last_face_seen_time = time.time()
                            self.cap = probe_cap
                            continue
                    probe_cap.release()
                time.sleep(1.0)
                continue

            # --- CAMERA ACQUISITION ---
            if self.cap is None or not self.cap.isOpened():
                self.cap = self.open_camera()
                if self.cap is None:
                    if not self.camera_busy:
                        print("[-] Camera is currently unavailable or used by another application. Waiting to auto-reconnect...")
                        self.db.log_event("Camera Status", "Camera locked by another app / disconnected")
                        self.camera_busy = True
                    time.sleep(self.retry_interval)
                    continue
                else:
                    if self.camera_busy:
                        print("[+] Camera reacquired successfully! Resuming tracking.")
                        self.db.log_event("Camera Status", "Camera reacquired and tracking resumed")
                        self.camera_busy = False

            # --- FRAME READ & MIRROR ---
            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures > 15:
                    print("[-] Camera lock lost (another tool opened the camera). Releasing handle...")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    self.camera_busy = True
                    consecutive_read_failures = 0
                    time.sleep(self.retry_interval)
                else:
                    time.sleep(0.03)
                continue
            
            consecutive_read_failures = 0
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            # In background headless mode: enforce 1.0 second sleep cadence to save CPU
            if self.headless and (now - last_process_time < self.sample_interval_sec):
                time.sleep(0.1)
                continue

            last_process_time = now

            # --- AI PROCESSING PIPELINE ---
            results = self.face_eng.process_frame(frame)
            
            bpm = None
            filtered_signal = []
            emotion = "No Face"
            emotion_conf = 0.0
            ear_avg = 0.0
            mar = 0.0
            pitch, yaw, roll, focus = 0.0, 0.0, 0.0, 0.0
            fatigue = "No Face"
            stress_level = "Unknown"
            stress_score = 0.0

            if results.multi_face_landmarks:
                self.last_face_seen_time = time.time()
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 1. 3D Head Pose + Iris Gaze Fusion
                pitch, yaw, roll, focus, gaze_x, gaze_y = self.face_eng.estimate_head_pose_and_focus(landmarks, w, h)
                
                # 2. EAR & MAR
                ear_l = self.health.calculate_ear(landmarks, self.health.LEFT_EYE_IDX, w, h)
                ear_r = self.health.calculate_ear(landmarks, self.health.RIGHT_EYE_IDX, w, h)
                mar = self.health.calculate_mar(landmarks, w, h)
                
                # 3. ISO PERCLOS & Blink Dynamics
                ear_avg, mar, perclos, blinks_per_min, fatigue, is_drowsy = self.health.update_fatigue_and_blinks(
                    ear_l, ear_r, mar
                )
                if is_drowsy:
                    self.db.log_event("Drowsiness Alert", f"PERCLOS reached {perclos:.1f}% or extended eye closure")

                # 4. rPPG Vitals
                raw_color = self.health.extract_forehead_roi(frame, landmarks, w, h)
                bpm, filtered_signal = self.health.update_rppg(raw_color)

                # 5. AffectNet Emotion Classification
                emotion, emotion_conf, valence = self.emotion_eng.analyze_emotion(
                    frame, landmarks, ear_avg, mar, pitch, w, h
                )

                # 6. Multimodal Stress Index (MSI: 0 - 100)
                stress_score, stress_level = self.health.compute_multimodal_stress(
                    bpm, blinks_per_min, emotion, pitch, yaw
                )

                # 7. High Stress Alert Detection
                if stress_score > 65.0:
                    self.high_stress_count += 1
                    if self.high_stress_count >= self.high_stress_threshold:
                        alert_triggered = self.notifier.trigger_stress_alert(
                            bpm=bpm, 
                            reason=f"Multimodal Stress Score reached {stress_score:.0f}/100"
                        )
                        if alert_triggered:
                            self.db.log_event("High Stress Alert", f"Stress popup shown (MSI: {stress_score:.1f}, BPM: {bpm})")
                        self.high_stress_count = 0
                else:
                    self.high_stress_count = max(0, self.high_stress_count - 1)

                # 8. Log to SQLite (1 Hz)
                self.db.log_metric(bpm, emotion, emotion_conf, ear_avg, mar, pitch, yaw, roll, focus, fatigue, f"{stress_level} ({stress_score:.0f})")

            else:
                # --- 15-MINUTE INACTIVITY CHECK ---
                inactive_duration = time.time() - self.last_face_seen_time
                if inactive_duration >= self.no_face_timeout_sec:
                    print(f"\n[!] No face detected for {inactive_duration/60.0:.1f} minutes. Triggering automatic log organization...")
                    
                    finished_session_id = self.db.end_session()
                    if finished_session_id:
                        self.organizer.organize_session(finished_session_id)

                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    self.in_standby = True
                    print("[i] Entered low-power standby. Will automatically wake up when you return.")
                    continue

            # Optional GUI display when testing with --gui
            if not self.headless:
                from main import draw_hud
                draw_hud(frame, bpm, filtered_signal, emotion, emotion_conf, ear_avg, mar, pitch, yaw, roll, focus, fatigue, f"{stress_level} ({stress_score:.0f})", self.session_id)
                if not results.multi_face_landmarks:
                    away_sec = int(time.time() - self.last_face_seen_time)
                    remaining = max(0, self.no_face_timeout_sec - away_sec)
                    cv2.putText(frame, f"Away: {away_sec}s | Standby in: {remaining}s", (w // 2 - 170, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                
                cv2.imshow("AI Real-Time Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Background mode sleep between samples
                time.sleep(self.sample_interval_sec)

        # Cleanup
        if self.cap:
            self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        final_id = self.db.end_session()
        if final_id:
            self.organizer.organize_session(final_id)
        print("[+] Tracker Service terminated.")

    def stop(self):
        self.running = False

if __name__ == "__main__":
    is_gui = "--gui" in sys.argv
    service = BackgroundTrackerService(headless=not is_gui, no_face_timeout_sec=900, sample_interval_sec=1.0)
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
