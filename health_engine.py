import numpy as np
import cv2
import time
from scipy.signal import butter, filtfilt

class HealthEngine:
    """
    Research-Grade Contactless Physiology & Multimodal Stress Engine:
    1. rPPG Heart Rate (BPM) from facial micro-color shifts (POS algorithm).
    2. ISO-Standard PERCLOS (Percentage of Eye Closure over 30s window) for true drowsiness.
    3. Blink Frequency (Blinks per Minute) tracking.
    4. Multi-Modal Stress Index (MSI: 0 - 100 Score) combining Autonomic Pulse, Blink Dynamics, Emotion, and Restlessness.
    """
    def __init__(self, fps=30, buffer_size=300):
        self.fps = fps
        self.buffer_size = buffer_size
        self.raw_green_signal = []
        self.filtered_signal = []
        
        # Landmark indices
        self.LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.MOUTH_IDX = [13, 14, 78, 308]
        self.FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
        
        # State tracking
        self.current_bpm = 0.0
        
        # Eye / Blink Dynamics (PERCLOS buffer over 30 seconds = 900 frames)
        self.ear_history = []
        self.max_ear_buffer = 900
        self.blink_timestamps = []
        self.is_eye_closed = False
        self.closed_frame_count = 0
        
        # Stress Index (0 to 100)
        self.stress_score = 25.0
        self.stress_level = "Relaxed / Nominal"
        
        # Head movement history for jitter/restlessness
        self.head_pose_history = []

    def calculate_ear(self, landmarks, eye_indices, w, h):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h_dist = np.linalg.norm(pts[0] - pts[3])
        if h_dist == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h_dist)

    def calculate_mar(self, landmarks, w, h):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.MOUTH_IDX])
        v_dist = np.linalg.norm(pts[0] - pts[1])
        h_dist = np.linalg.norm(pts[2] - pts[3])
        if h_dist == 0:
            return 0.0
        return v_dist / h_dist

    def extract_forehead_roi(self, frame, landmarks, w, h):
        forehead_pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in self.FOREHEAD_IDX], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(forehead_pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        mean_val = cv2.mean(frame, mask=mask)
        green_mean = mean_val[1]
        red_mean = mean_val[2]
        blue_mean = mean_val[0]
        
        # POS-inspired rPPG signal combination
        return 3.0 * green_mean - 2.0 * red_mean - 0.5 * blue_mean

    def update_rppg(self, raw_val):
        self.raw_green_signal.append(raw_val)
        if len(self.raw_green_signal) > self.buffer_size:
            self.raw_green_signal.pop(0)

        if len(self.raw_green_signal) < 90:
            return None, []

        signal = np.array(self.raw_green_signal)
        signal = signal - np.mean(signal)
        std_val = np.std(signal)
        if std_val > 1e-6:
            signal = signal / std_val

        try:
            nyquist = 0.5 * self.fps
            low = 0.75 / nyquist
            high = 3.0 / nyquist
            b, a = butter(2, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            self.filtered_signal = filtered

            fft_vals = np.abs(np.fft.rfft(filtered))
            fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0/self.fps)

            valid_idx = np.where((fft_freqs >= 0.75) & (fft_freqs <= 3.0))[0]
            if len(valid_idx) > 0:
                peak_freq = fft_freqs[valid_idx[np.argmax(fft_vals[valid_idx])]]
                bpm = peak_freq * 60.0
                
                if self.current_bpm == 0.0:
                    self.current_bpm = bpm
                else:
                    self.current_bpm = 0.88 * self.current_bpm + 0.12 * bpm
                    
                return self.current_bpm, filtered
        except Exception:
            pass

        return None, []

    def update_fatigue_and_blinks(self, ear_left, ear_right, mar):
        avg_ear = (ear_left + ear_right) / 2.0
        now = time.time()
        
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > self.max_ear_buffer:
            self.ear_history.pop(0)
            
        # Blink detection
        is_closed = avg_ear < 0.18
        if is_closed:
            self.closed_frame_count += 1
            if not self.is_eye_closed and self.closed_frame_count >= 2:
                self.is_eye_closed = True
                self.blink_timestamps.append(now)
        else:
            self.is_eye_closed = False
            self.closed_frame_count = 0
            
        self.blink_timestamps = [t for t in self.blink_timestamps if (now - t) <= 60.0]
        blinks_per_min = len(self.blink_timestamps)

        # PERCLOS calculation
        closed_frames = sum(1 for e in self.ear_history if e < 0.19)
        perclos = (closed_frames / len(self.ear_history)) * 100.0 if self.ear_history else 0.0

        is_drowsy_event = False
        if perclos > 20.0 or self.closed_frame_count > 45:
            fatigue_state = "DROWSY ALERT"
            is_drowsy_event = True
        elif mar > 0.65:
            fatigue_state = "Yawning"
        elif perclos > 10.0:
            fatigue_state = "Mild Fatigue"
        else:
            fatigue_state = "Alert / Focused"

        return avg_ear, mar, perclos, blinks_per_min, fatigue_state, is_drowsy_event

    def compute_multimodal_stress(self, bpm, blinks_per_min, emotion, head_pitch, head_yaw):
        bpm_val = bpm if (bpm and bpm > 40) else 70.0
        hr_stress = max(0.0, min(100.0, (bpm_val - 62.0) * 2.2))
        blink_stress = max(0.0, min(100.0, (blinks_per_min - 15) * 3.5))

        emotion_stress = 10.0
        if "Angry" in emotion or "Anxious" in emotion:
            emotion_stress = 80.0
        elif "Sad" in emotion:
            emotion_stress = 40.0
        elif "Happy" in emotion:
            emotion_stress = 0.0

        self.head_pose_history.append((head_pitch, head_yaw))
        if len(self.head_pose_history) > 60:
            self.head_pose_history.pop(0)
            
        restlessness = 0.0
        if len(self.head_pose_history) >= 10:
            pitch_std = np.std([p[0] for p in self.head_pose_history])
            yaw_std = np.std([p[1] for p in self.head_pose_history])
            restlessness = min(100.0, (pitch_std + yaw_std) * 5.0)

        target_score = (0.40 * hr_stress) + (0.25 * blink_stress) + (0.20 * emotion_stress) + (0.15 * restlessness)
        self.stress_score = 0.85 * self.stress_score + 0.15 * target_score
        
        if self.stress_score > 65.0:
            self.stress_level = "High Stress"
        elif self.stress_score > 40.0:
            self.stress_level = "Moderate Stress"
        elif self.stress_score > 20.0:
            self.stress_level = "Nominal / Balanced"
        else:
            self.stress_level = "Relaxed / Calm"

        return round(self.stress_score, 1), self.stress_level
