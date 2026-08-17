import sqlite3
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class ReportOrganizer:
    """
    Analyzes, summarizes, and organizes raw session logs into structured reports,
    daily summaries, and JSON/CSV archives when a session concludes (e.g. after user leaves).
    """
    def __init__(self, db_path="tracker_history.db", output_dir="session_reports"):
        self.db_path = db_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def organize_session(self, session_id):
        """Generates a complete organized analytical summary for the concluded session."""
        if not os.path.exists(self.db_path):
            return None

        conn = sqlite3.connect(self.db_path)
        metrics_df = pd.read_sql_query("SELECT * FROM metrics WHERE session_id = ?", conn, params=(session_id,))
        session_df = pd.read_sql_query("SELECT * FROM sessions WHERE id = ?", conn, params=(session_id,))
        events_df = pd.read_sql_query("SELECT * FROM events WHERE session_id = ?", conn, params=(session_id,))
        conn.close()

        if metrics_df.empty or session_df.empty:
            return None

        # 1. Compute Deep Analytics
        bpm_data = metrics_df['heart_rate_bpm'].dropna()
        avg_bpm = float(bpm_data.mean()) if not bpm_data.empty else None
        min_bpm = float(bpm_data.min()) if not bpm_data.empty else None
        max_bpm = float(bpm_data.max()) if not bpm_data.empty else None
        
        # Emotion breakdown percentages
        emotion_counts = metrics_df['primary_emotion'].value_counts(normalize=True) * 100
        emotion_breakdown = {str(k): round(float(v), 1) for k, v in emotion_counts.items()}
        
        # Focus & Fatigue statistics
        avg_focus = float(metrics_df['focus_score'].mean()) if 'focus_score' in metrics_df else 0.0
        avg_ear = float(metrics_df['ear'].mean()) if 'ear' in metrics_df else 0.0
        drowsy_events = len(events_df[events_df['event_type'].str.contains('Drowsy', case=False, na=False)])
        
        duration = float(session_df['duration_seconds'].iloc[0]) if 'duration_seconds' in session_df and not pd.isna(session_df['duration_seconds'].iloc[0]) else len(metrics_df)
        start_time = session_df['start_time'].iloc[0]
        end_time = session_df['end_time'].iloc[0] if 'end_time' in session_df else datetime.now().isoformat()

        report = {
            "session_id": session_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": round(duration / 60.0, 2),
            "vitals_summary": {
                "avg_heart_rate_bpm": round(avg_bpm, 1) if avg_bpm else "N/A",
                "min_heart_rate_bpm": round(min_bpm, 1) if min_bpm else "N/A",
                "max_heart_rate_bpm": round(max_bpm, 1) if max_bpm else "N/A"
            },
            "affective_summary": {
                "dominant_emotion": session_df['dominant_emotion'].iloc[0] if not pd.isna(session_df['dominant_emotion'].iloc[0]) else "Unknown",
                "emotion_breakdown_percent": emotion_breakdown
            },
            "engagement_and_fatigue": {
                "average_focus_score_pct": round(avg_focus, 1),
                "average_eye_openness_ear": round(avg_ear, 3),
                "drowsiness_alerts_count": drowsy_events
            }
        }

        # 2. Save JSON analytical report
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"session_{session_id}_{timestamp_slug}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        # 3. Export specific session metrics to CSV
        csv_path = os.path.join(self.output_dir, f"session_{session_id}_{timestamp_slug}_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)

        print("\n" + "="*70)
        print(f"       ORGANIZED LOG REPORT (Session #{session_id})")
        print("="*70)
        print(f" Duration:            {report['duration_minutes']} minutes")
        print(f" Avg Heart Rate:      {report['vitals_summary']['avg_heart_rate_bpm']} BPM")
        print(f" Dominant Emotion:    {report['affective_summary']['dominant_emotion']}")
        print(f" Avg Focus Score:     {report['engagement_and_fatigue']['average_focus_score_pct']}%")
        print(f" Drowsiness Alerts:   {report['engagement_and_fatigue']['drowsiness_alerts_count']}")
        print(f" Report JSON:         {json_path}")
        print(f" Data CSV:            {csv_path}")
        print("="*70 + "\n")

        return report
