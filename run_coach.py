import sys
import os
from ai_coach import LocalLLMHealthCoach
from notifier import NotificationManager

def run_coach(show_popup=True):
    print("=======================================================")
    print("       AI HEALTH COACH: LOCAL LLM DAILY INSIGHTS       ")
    print("=======================================================\n")
    
    coach = LocalLLMHealthCoach()
    is_online, models = coach.is_ollama_available()
    if is_online:
        print(f"[+] Local LLM Engine (Ollama): ONLINE (Models detected: {', '.join(models)})")
    else:
        print("[i] Local LLM Engine (Ollama): Standby / Offline (Using intelligent local rule engine)")
        print("    Tip: Run `ollama run llama3` or `ollama run gemma2` anytime for deep LLM synthesis!\n")

    report = coach.generate_daily_coaching_report()
    print("-------------------------------------------------------")
    print(report)
    print("-------------------------------------------------------\n")

    if show_popup:
        notifier = NotificationManager()
        notifier.show_daily_coaching_popup(report)
        print("[+] Coaching summary popup displayed on your desktop!")

if __name__ == "__main__":
    run_coach(show_popup=True)
