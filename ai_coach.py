import json
import os
import sqlite3
import pandas as pd
import requests
from datetime import datetime

class LocalLLMHealthCoach:
    """
    AI Health & Wellness Coach powered by Local LLMs (Ollama / Llama 3 / Gemma 2 / Mistral / Qwen).
    - Ingests daily session analytics from SQLite / session_reports.
    - Generates personalized, empathetic, research-backed daily insights & recommendations.
    - Operates 100% locally on your machine with zero cloud API keys or data leakage.
    - Features smart rule-based fallback if Ollama/local LLM is not currently running.
    """
    def __init__(self, ollama_url="http://localhost:11434", model="llama3:latest", db_path="tracker_history.db"):
        self.ollama_url = ollama_url
        self.model = model
        self.db_path = db_path

    def is_ollama_available(self):
        """Checks if local Ollama service is running on the machine."""
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            if res.status_code == 200:
                models = [m['name'] for m in res.json().get('models', [])]
                return True, models
        except Exception:
            pass
        return False, []

    def get_today_summary_data(self):
        """Aggregates all sessions and metrics recorded today."""
        if not os.path.exists(self.db_path):
            return None

        conn = sqlite3.connect(self.db_path)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Pull today's sessions
        sessions_df = pd.read_sql_query(
            "SELECT * FROM sessions WHERE start_time LIKE ? ORDER BY id ASC", 
            conn, params=(f"{today_str}%",)
        )
        
        # Pull today's metrics
        metrics_df = pd.read_sql_query(
            "SELECT * FROM metrics WHERE timestamp LIKE ? ORDER BY id ASC", 
            conn, params=(f"{today_str}%",)
        )
        
        # Pull today's events
        events_df = pd.read_sql_query(
            "SELECT * FROM events WHERE timestamp LIKE ? ORDER BY id ASC", 
            conn, params=(f"{today_str}%",)
        )
        conn.close()

        if metrics_df.empty:
            return None

        # Analytics calculations
        total_time_min = round(len(metrics_df) / 60.0, 1)
        avg_bpm = round(float(metrics_df['heart_rate_bpm'].dropna().mean()), 1) if not metrics_df['heart_rate_bpm'].dropna().empty else "N/A"
        max_bpm = round(float(metrics_df['heart_rate_bpm'].dropna().max()), 1) if not metrics_df['heart_rate_bpm'].dropna().empty else "N/A"
        avg_focus = round(float(metrics_df['focus_score'].mean()), 1) if 'focus_score' in metrics_df else 0.0
        
        # Emotion distribution
        emotion_counts = metrics_df['primary_emotion'].value_counts(normalize=True) * 100
        emotion_breakdown = {str(k): f"{round(float(v), 1)}%" for k, v in emotion_counts.items()}
        
        # High stress intervals
        high_stress_rows = metrics_df[metrics_df['stress_index'].str.contains('High', case=False, na=False)]
        stress_events_count = len(events_df[events_df['event_type'].str.contains('Stress', case=False, na=False)])
        drowsy_events_count = len(events_df[events_df['event_type'].str.contains('Drowsy', case=False, na=False)])

        return {
            "date": today_str,
            "total_desk_time_minutes": total_time_min,
            "sessions_count": len(sessions_df),
            "avg_heart_rate_bpm": avg_bpm,
            "max_heart_rate_bpm": max_bpm,
            "average_screen_focus_score": f"{avg_focus}%",
            "emotion_breakdown": emotion_breakdown,
            "high_stress_occurrences": stress_events_count,
            "drowsiness_alerts": drowsy_events_count,
        }

    def generate_daily_coaching_report(self):
        """Generates AI coaching insight using Local LLM or Intelligent Fallback."""
        data = self.get_today_summary_data()
        if not data:
            return "No tracking data recorded for today yet. Use your computer with the tracker running to generate insights!"

        ollama_online, available_models = self.is_ollama_available()

        if ollama_online:
            # Select available model (prefer llama3, gemma2, mistral, or first available)
            selected_model = self.model
            if available_models:
                for cand in ['llama3', 'gemma2', 'mistral', 'qwen2', 'phi3']:
                    matched = [m for m in available_models if cand in m.lower()]
                    if matched:
                        selected_model = matched[0]
                        break
                else:
                    selected_model = available_models[0]

            prompt = (
                f"You are an empathetic, scientific AI Health & Ergonomics Coach. "
                f"Analyze the user's computer workstation health data for today:\n\n"
                f"{json.dumps(data, indent=2)}\n\n"
                f"Instructions:\n"
                f"1. Summarize their focus, autonomic stress, and energy levels in 2-3 friendly sentences.\n"
                f"2. Point out specific patterns (e.g. peak stress or fatigue spikes).\n"
                f"3. Provide exactly 2 actionable, personalized tips for tomorrow (e.g., break timing, eye relaxation, posture).\n"
                f"Keep your tone supportive, concise, and motivational (under 150 words)."
            )

            try:
                payload = {
                    "model": selected_model,
                    "prompt": prompt,
                    "stream": False
                }
                res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=25.0)
                if res.status_code == 200:
                    llm_response = res.json().get("response", "").strip()
                    if llm_response:
                        return f"🤖 AI Coach Insight ({selected_model}):\n\n{llm_response}"
            except Exception as e:
                pass

        # Intelligent Rule-Based Fallback (Works instantly without requiring Ollama)
        return self._generate_rule_based_coaching(data)

    def _generate_rule_based_coaching(self, data):
        focus = float(data['average_screen_focus_score'].replace('%', ''))
        bpm = data['avg_heart_rate_bpm']
        stress_count = data['high_stress_occurrences']
        drowsy_count = data['drowsiness_alerts']
        total_time = data['total_desk_time_minutes']
        
        insights = []
        insights.append(f"📊 Today's Health Summary ({data['date']}):")
        insights.append(f"• Active Desk Time: {total_time} mins | Average Focus: {focus}% | Avg Heart Rate: {bpm} BPM")
        
        if stress_count > 0:
            insights.append(f"• Stress Pattern: You experienced {stress_count} elevated stress intervals today. Consider micro-breaks before high-intensity focus blocks.")
        else:
            insights.append("• Autonomic Balance: Your heart rate and facial tension remained steady and relaxed today. Great job!")

        if drowsy_count > 0:
            insights.append(f"• Fatigue Alert: {drowsy_count} moments of slow eye closure were recorded. Ensure proper lighting and 7+ hours of sleep.")
        elif focus > 80:
            insights.append("• High Engagement: Excellent sustained focus throughout your working sessions.")

        insights.append("\n💡 Coach Recommendations for Tomorrow:")
        insights.append("1. Practice the 20-20-20 Rule: Every 20 minutes, gaze 20 feet away for 20 seconds to prevent digital eye fatigue.")
        insights.append("2. Optimal Break Timing: Take a 5-minute movement or hydration break every 50 minutes of deep work.")

        return "\n".join(insights)
