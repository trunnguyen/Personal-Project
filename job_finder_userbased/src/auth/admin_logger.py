import sqlite3
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
ADMIN_DB_PATH = os.path.join(PROJECT_ROOT, "data", "admin.db")


def init_admin_db():
    os.makedirs(os.path.dirname(ADMIN_DB_PATH), exist_ok=True)
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                email_hash TEXT,
                detail TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                email_hash TEXT,
                action TEXT,
                detail TEXT
            )
        ''')


def log_security_event(event_type, email_hash, detail=""):
    init_admin_db()
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO security_events (event_type, email_hash, detail) VALUES (?, ?, ?)",
            (event_type, email_hash, detail)
        )
        conn.commit()


def log_user_activity(email_hash, action, detail=""):
    init_admin_db()
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO user_activity (email_hash, action, detail) VALUES (?, ?, ?)",
            (email_hash, action, detail)
        )
        conn.commit()


def get_recent_security_events(limit=50):
    init_admin_db()
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM security_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_recent_user_activity(limit=50):
    init_admin_db()
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM user_activity ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]