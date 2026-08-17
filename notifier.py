import threading
import time
import ctypes
import os

class NotificationManager:
    """
    Handles non-blocking, asynchronous Windows desktop alert dialogs and toast-like
    notifications with health and stress-relief suggestions.
    """
    def __init__(self, cooldown_seconds=180):
        # 3-minute cooldown between alerts to avoid spamming while user is working
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0
        self.active_alert = False

    def trigger_stress_alert(self, bpm=None, reason="Elevated heart rate & facial tension"):
        """Triggers a non-blocking Windows desktop message box with actionable stress-reduction tips."""
        current_time = time.time()
        if (current_time - self.last_alert_time < self.cooldown_seconds) or self.active_alert:
            return False

        self.last_alert_time = current_time
        
        # Run alert in a detached background thread so camera tracking never pauses
        thread = threading.Thread(target=self._show_dialog, args=(bpm, reason), daemon=True)
        thread.start()
        return True

    def _show_dialog(self, bpm, reason):
        self.active_alert = True
        bpm_str = f"Estimated Heart Rate: {bpm:.0f} BPM\n" if bpm else ""
        
        title = "AI Health Alert: Elevated Stress Detected"
        message = (
            f"⚠️ High Stress Level Detected!\n\n"
            f"{bpm_str}"
            f"Trigger: {reason}\n\n"
            f"Recommended Immediate Actions:\n"
            f"1. 🧘 Take 3 Deep Breaths: Inhale for 4s, Hold for 4s, Exhale for 6s.\n"
            f"2. 💧 Drink a glass of water and rest your eyes from the screen.\n"
            f"3. 🙆 Gently stretch your neck and shoulders to release facial tension.\n\n"
            f"Click OK to dismiss."
        )
        
        # MB_OK | MB_ICONWARNING | MB_TOPMOST (0x00000000 | 0x00000030 | 0x00040000)
        MB_ICONWARNING = 0x30
        MB_TOPMOST = 0x40000
        ctypes.windll.user32.MessageBoxW(0, message, title, MB_ICONWARNING | MB_TOPMOST)
        self.active_alert = False

    def show_daily_coaching_popup(self, coaching_text):
        """Displays a dedicated AI Health Coach Insight popup window."""
        thread = threading.Thread(target=self._show_coach_dialog, args=(coaching_text,), daemon=True)
        thread.start()

    def _show_coach_dialog(self, coaching_text):
        title = "🌿 AI Health & Performance Coach: Daily Insights"
        # MB_OK | MB_ICONINFORMATION | MB_TOPMOST (0x00000000 | 0x00000040 | 0x00040000)
        MB_ICONINFORMATION = 0x40
        MB_TOPMOST = 0x40000
        ctypes.windll.user32.MessageBoxW(0, coaching_text, title, MB_ICONINFORMATION | MB_TOPMOST)
