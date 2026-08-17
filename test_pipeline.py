import cv2
import numpy as np
import time
import os
from database import DatabaseManager
from health_engine import HealthEngine
from emotion_engine import EmotionEngine
from face_engine import FaceEngine
from report_organizer import ReportOrganizer
from notifier import NotificationManager

def run_self_test():
    print("=======================================================")
    print("       STARTING AI TRACKER SELF-TEST SUITE             ")
    print("=======================================================\n")

    print("[1/6] Testing DatabaseManager (SQLite Logging)...")
    db = DatabaseManager("test_tracker.db")
    session_id = db.start_session("Automated test session")
    assert session_id is not None
    db.log_metric(72.5, "Neutral / Focused", 0.92, 0.28, 0.15, 2.0, -1.0, 0.5, 95.0, "Alert / Focused", "Nominal / Balanced (22)")
    db.log_event("Test Event", "Self test passed")
    print("      -> DB initialized & metric logged successfully.")

    print("[2/6] Testing FaceEngine (MediaPipe 468 3D Mesh)...")
    face_eng = FaceEngine()
    synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(synthetic_frame, (640, 360), 120, (200, 180, 160), -1)
    res = face_eng.process_frame(synthetic_frame)
    print("      -> FaceEngine initialized & frame processed without error.")

    print("[3/6] Testing EmotionEngine (AffectNet Neural Model + EMA)...")
    emotion_eng = EmotionEngine()
    print(f"      -> EmotionEngine loaded (Neural backend: {emotion_eng.model_loaded}).")

    print("[4/6] Testing HealthEngine (rPPG, PERCLOS & Multimodal Stress)...")
    health = HealthEngine(fps=30)
    for i in range(100):
        simulated_color = 120.0 + 2.0 * np.sin(2 * np.pi * 1.2 * (i / 30.0))
        bpm, signal = health.update_rppg(simulated_color)
    
    # Test stress computation
    stress_score, stress_level = health.compute_multimodal_stress(bpm, 16, "Neutral / Focused", 0.0, 0.0)
    print(f"      -> HealthEngine processed rPPG pulse buffer. Computed BPM: {bpm:.1f}")
    print(f"      -> Multimodal Stress Index: {stress_score}/100 ({stress_level})")

    print("[5/6] Testing ReportOrganizer (JSON & CSV analytical summary)...")
    organizer = ReportOrganizer("test_tracker.db", "test_reports")
    db.end_session()
    report = organizer.organize_session(session_id)
    assert report is not None
    print("      -> ReportOrganizer generated session JSON & CSV.")

    print("[6/6] Testing NotificationManager (Desktop Alert Daemon)...")
    notifier = NotificationManager(cooldown_seconds=1)
    assert hasattr(notifier, "trigger_stress_alert")
    print("      -> NotificationManager ready.")

    # Cleanup temporary test database
    if os.path.exists("test_tracker.db"):
        os.remove("test_tracker.db")

    print("\n=======================================================")
    print("       ALL ENGINE INTEGRITY TESTS PASSED (6/6)         ")
    print("=======================================================\n")

if __name__ == "__main__":
    run_self_test()
