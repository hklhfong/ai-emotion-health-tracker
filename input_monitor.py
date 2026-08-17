import ctypes
from ctypes import Structure, c_uint, sizeof, byref

class LASTINPUTINFO(Structure):
    _fields_ = [
        ('cbSize', c_uint),
        ('dwTime', c_uint)
    ]

def get_idle_duration_seconds():
    """
    Returns the number of seconds since the last physical user input
    (mouse movement, mouse click, or keyboard stroke) on Windows using GetLastInputInfo.
    Zero external dependencies, ultra-fast and native.
    """
    last_input = LASTINPUTINFO()
    last_input.cbSize = sizeof(LASTINPUTINFO)
    if ctypes.windll.user32.GetLastInputInfo(byref(last_input)):
        millis_since_boot = ctypes.windll.kernel32.GetTickCount()
        elapsed_millis = millis_since_boot - last_input.dwTime
        return max(0.0, elapsed_millis / 1000.0)
    return 0.0

def wait_for_user_activity(poll_interval=0.5):
    """
    Blocks efficiently until mouse movement or keyboard press is detected.
    """
    initial_idle = get_idle_duration_seconds()
    while True:
        import time
        time.sleep(poll_interval)
        current_idle = get_idle_duration_seconds()
        # If the idle timer reset to near zero (< 0.6s), the user moved mouse or typed
        if current_idle < 0.6 or current_idle < initial_idle:
            return True
