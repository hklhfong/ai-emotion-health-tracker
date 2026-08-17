import threading
import time
import os
import sys
from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item
from service import BackgroundTrackerService
from ai_coach import LocalLLMHealthCoach
from notifier import NotificationManager

class SystemTrayApp:
    """
    Windows System Tray Controller for AI Emotion, Health & Face Tracker.
    Features:
    - Runs quietly in Windows Notification Area (Taskbar Tray).
    - Status Badge: Active 🟢 / Paused ⏸️ / Camera Busy 📷
    - Right-Click Context Menu:
        • Pause / Resume Tracking
        • Trigger Daily AI Health Coach (Ollama / Local LLM)
        • Open Today's Analytical Reports Folder
        • View SQLite Database Summary
        • Launch Interactive Visual HUD Window
        • Exit Application
    """
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        self.service = BackgroundTrackerService(headless=True, sample_interval_sec=1.0)
        self.service_thread = None
        self.is_paused = False
        self.icon = None
        self.notifier = NotificationManager()
        self.coach = LocalLLMHealthCoach()

    def create_tray_icon_image(self, color="green"):
        """Generates a modern circular tray icon with status color."""
        width = 64
        height = 64
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Outer ring
        draw.ellipse((4, 4, width - 4, height - 4), outline=(40, 40, 40), width=3)
        
        # Inner fill color based on state
        fill_color = (0, 220, 100) if color == "green" else ((255, 180, 0) if color == "yellow" else (220, 50, 50))
        draw.ellipse((14, 14, width - 14, height - 14), fill=fill_color)
        
        # Inner pulse dot
        draw.ellipse((26, 26, width - 26, height - 26), fill=(255, 255, 255))
        return image

    def start_background_service(self):
        """Runs the background tracking loop in a dedicated worker thread."""
        self.service_thread = threading.Thread(target=self.service.start, daemon=True)
        self.service_thread.start()

    def toggle_pause(self, icon, item):
        """Pause / Resume toggle."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("[i] Tracking Paused by user.")
            self.service.stop()
            self.icon.icon = self.create_tray_icon_image("yellow")
            self.icon.title = "AI Health Tracker: Paused ⏸️"
        else:
            print("[+] Tracking Resumed.")
            self.service = BackgroundTrackerService(headless=True, sample_interval_sec=1.0)
            self.start_background_service()
            self.icon.icon = self.create_tray_icon_image("green")
            self.icon.title = "AI Health Tracker: Active 🟢"

    def trigger_ai_coach(self, icon, item):
        """Generates and displays AI Coach insights in a background thread."""
        def _task():
            report = self.coach.generate_daily_coaching_report()
            self.notifier.show_daily_coaching_popup(report)
        threading.Thread(target=_task, daemon=True).start()

    def open_reports_folder(self, icon, item):
        """Opens the session_reports directory in Windows File Explorer."""
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        os.startfile(reports_dir)

    def launch_interactive_hud(self, icon, item):
        """Launches the visual HUD testing window."""
        bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_tracker.bat")
        os.startfile(bat_path)

    def exit_app(self, icon, item):
        """Clean shutdown."""
        print("[+] Shutting down AI Tracker Tray App...")
        self.service.stop()
        self.icon.stop()
        sys.exit(0)

    def run(self):
        # Start background worker
        self.start_background_service()

        # Build System Tray Menu
        menu = pystray.Menu(
            item("🟢 Status: Tracking Active (1 Hz)", lambda icon, item: None, enabled=False),
            pystray.Menu.SEPARATOR,
            item(lambda text: "⏸️ Pause Tracking" if not self.is_paused else "▶️ Resume Tracking", self.toggle_pause),
            item("🤖 Ask AI Health Coach (Ollama)", self.trigger_ai_coach),
            item("📁 Open Session Reports (CSV/JSON)", self.open_reports_folder),
            item("🖥️ Launch Live Visual HUD", self.launch_interactive_hud),
            pystray.Menu.SEPARATOR,
            item("❌ Exit AI Tracker", self.exit_app)
        )

        self.icon = pystray.Icon(
            "AITracker",
            self.create_tray_icon_image("green"),
            "AI Emotion, Health & Face Tracker (Active)",
            menu
        )

        print("[+] System Tray App running. Look for the green dot in your Windows Taskbar tray!")
        self.icon.run()

if __name__ == "__main__":
    app = SystemTrayApp()
    app.run()
