import sqlite3
import pandas as pd
import sys
import os

DB_PATH = "tracker_history.db"

def show_all_sessions():
    if not os.path.exists(DB_PATH):
        print("[-] No database found yet. Run main.py to record your first session!")
        return

    conn = sqlite3.connect(DB_PATH)
    sessions_df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", conn)
    conn.close()
    
    print("\n" + "="*80)
    print("                      TRACKING SESSIONS HISTORY")
    print("="*80)
    if sessions_df.empty:
        print("No sessions recorded yet.")
    else:
        print(sessions_df.to_string(index=False))
    print("="*80 + "\n")

def export_all():
    if not os.path.exists(DB_PATH):
        print("[-] No database found yet.")
        return

    conn = sqlite3.connect(DB_PATH)
    metrics_df = pd.read_sql_query("SELECT * FROM metrics", conn)
    sessions_df = pd.read_sql_query("SELECT * FROM sessions", conn)
    events_df = pd.read_sql_query("SELECT * FROM events", conn)
    conn.close()

    metrics_df.to_csv("all_metrics_history.csv", index=False)
    sessions_df.to_csv("all_sessions_summary.csv", index=False)
    events_df.to_csv("all_events_history.csv", index=False)
    print("[+] Exported:")
    print("    - all_metrics_history.csv")
    print("    - all_sessions_summary.csv")
    print("    - all_events_history.csv")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_all()
    else:
        show_all_sessions()
        print("Tip: Run `python export_history.py --export` to export all history to CSV files.")
