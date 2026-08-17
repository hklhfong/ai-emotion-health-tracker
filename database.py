import sqlite3
import time
import os
import json
import csv
from datetime import datetime

class DatabaseManager:
    """Manages local SQLite storage for all tracking sessions and time-series metrics."""
    
    def __init__(self, db_path="tracker_history.db"):
        self.db_path = db_path
        self.init_db()
        self.current_session_id = None
        self.last_log_time = 0
        self.log_interval = 1.0  # Log metrics every 1.0 second by default

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds REAL,
                avg_bpm REAL,
                dominant_emotion TEXT,
                drowsiness_events_count INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        # 2. Detailed Time-Series Metrics (1 Hz logging)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT NOT NULL,
                heart_rate_bpm REAL,
                primary_emotion TEXT,
                emotion_confidence REAL,
                ear REAL,
                mar REAL,
                head_pitch REAL,
                head_yaw REAL,
                head_roll REAL,
                focus_score REAL,
                fatigue_level TEXT,
                stress_index TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # 3. Discrete Events Table (e.g. Alerts, posture warnings)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        conn.commit()
        conn.close()

    def start_session(self, notes=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        cursor.execute("INSERT INTO sessions (start_time, notes) VALUES (?, ?)", (now_str, notes))
        conn.commit()
        self.current_session_id = cursor.lastrowid
        conn.close()
        return self.current_session_id

    def log_metric(self, bpm, emotion, emotion_conf, ear, mar, pitch, yaw, roll, focus, fatigue, stress, force=False):
        current_time = time.time()
        if not force and (current_time - self.last_log_time < self.log_interval):
            return  # Rate limit to avoid bloat
        
        self.last_log_time = current_time
        if self.current_session_id is None:
            self.start_session()
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO metrics (
                session_id, timestamp, heart_rate_bpm, primary_emotion, emotion_confidence,
                ear, mar, head_pitch, head_yaw, head_roll, focus_score, fatigue_level, stress_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.current_session_id, now_str, round(bpm, 1) if bpm else None,
            emotion, round(emotion_conf, 2), round(ear, 3), round(mar, 3),
            round(pitch, 1), round(yaw, 1), round(roll, 1),
            round(focus, 1), fatigue, stress
        ))
        conn.commit()
        conn.close()

    def log_event(self, event_type, details=""):
        if self.current_session_id is None:
            self.start_session()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        cursor.execute("INSERT INTO events (session_id, timestamp, event_type, details) VALUES (?, ?, ?, ?)",
                       (self.current_session_id, now_str, event_type, details))
        conn.commit()
        conn.close()

    def end_session(self):
        if self.current_session_id is None:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        
        # Calculate summary statistics
        cursor.execute("SELECT AVG(heart_rate_bpm) FROM metrics WHERE session_id = ? AND heart_rate_bpm IS NOT NULL", (self.current_session_id,))
        avg_bpm = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT primary_emotion, COUNT(*) as cnt FROM metrics WHERE session_id = ? GROUP BY primary_emotion ORDER BY cnt DESC LIMIT 1", (self.current_session_id,))
        dom_emotion_row = cursor.fetchone()
        dominant_emotion = dom_emotion_row[0] if dom_emotion_row else "Unknown"
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE session_id = ? AND event_type LIKE '%Drowsy%'", (self.current_session_id,))
        drowsy_count = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT start_time FROM sessions WHERE id = ?", (self.current_session_id,))
        start_time_row = cursor.fetchone()
        duration = 0.0
        if start_time_row:
            try:
                start_dt = datetime.fromisoformat(start_time_row[0])
                duration = (datetime.now() - start_dt).total_seconds()
            except Exception:
                pass

        cursor.execute("""
            UPDATE sessions 
            SET end_time = ?, duration_seconds = ?, avg_bpm = ?, dominant_emotion = ?, drowsiness_events_count = ?
            WHERE id = ?
        """, (now_str, round(duration, 1), round(avg_bpm, 1), dominant_emotion, drowsy_count, self.current_session_id))
        
        conn.commit()
        conn.close()
        saved_id = self.current_session_id
        self.current_session_id = None
        return saved_id

    def export_session_to_csv(self, session_id, output_path=None):
        if output_path is None:
            output_path = f"session_{session_id}_export.csv"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metrics WHERE session_id = ? ORDER BY id ASC", (session_id,))
        rows = cursor.fetchall()
        headers = [description[0] for description in cursor.description]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
            
        conn.close()
        return output_path
