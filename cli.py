"""
CLI Entry Points for ai-emotion-health-tracker.

These functions are registered as console_scripts in pyproject.toml
so that after `pip install ai-emotion-health-tracker`, users can run:

    ai-tracker         → Visual HUD window
    ai-tracker-tray    → Silent system tray background mode
    ai-tracker-coach   → Generate AI health coaching report
"""


def run_visual():
    """Launch the interactive visual HUD tracker window."""
    from main import main
    main()


def run_tray():
    """Launch the silent Windows system tray background tracker."""
    from tray_app import SystemTrayApp
    app = SystemTrayApp()
    app.run()


def run_coach():
    """Generate and display a daily AI health coaching report."""
    from run_coach import run_coach as _run_coach
    _run_coach(show_popup=True)


if __name__ == "__main__":
    run_visual()
