"""
TruthGuard AI - Database Module
Handles all SQLite database operations.
Tables: users, scan_history
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'truthguard.db')


def get_db():
    """Get database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    return conn


def init_db():
    """Initialize database and create all tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            last_login  TEXT
        )
    ''')

    # Scan history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            scan_type     TEXT    NOT NULL,  -- 'text' or 'image'
            input_data    TEXT,              -- headline/article text or filename
            result        TEXT    NOT NULL,  -- 'Fake', 'Real', 'AI-Generated', 'Manipulated'
            confidence    REAL    NOT NULL,  -- 0.0 to 100.0
            details       TEXT,              -- JSON with extra info
            scanned_at    TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("  ✓ Database initialized at:", DB_PATH)


# ─── USER OPERATIONS ──────────────────────────────────────────────────────────

def create_user(username, email, password_hash):
    """Insert new user. Returns user id or None if duplicate."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, datetime.now().isoformat())
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None  # Duplicate username or email


def get_user_by_username(username):
    """Fetch user dict by username."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email):
    """Fetch user dict by email."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    """Fetch user dict by id."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_last_login(user_id):
    """Update last login timestamp."""
    conn = get_db()
    conn.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(), user_id)
    )
    conn.commit()
    conn.close()


# ─── SCAN HISTORY OPERATIONS ──────────────────────────────────────────────────

def save_scan(user_id, scan_type, input_data, result, confidence, details=None):
    """Save a detection scan result."""
    import json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO scan_history
           (user_id, scan_type, input_data, result, confidence, details, scanned_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (
            user_id,
            scan_type,
            input_data[:500] if input_data else '',  # Truncate long text
            result,
            round(float(confidence), 1),
            json.dumps(details) if details else None,
            datetime.now().isoformat()
        )
    )
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id


def get_user_history(user_id, limit=50):
    """Get scan history for a user, newest first."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT * FROM scan_history
           WHERE user_id = ?
           ORDER BY scanned_at DESC
           LIMIT ?''',
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_stats(user_id):
    """Get aggregated stats for dashboard."""
    conn = get_db()
    cursor = conn.cursor()

    # Total scans
    cursor.execute("SELECT COUNT(*) as total FROM scan_history WHERE user_id=?", (user_id,))
    total = cursor.fetchone()['total']

    # By type
    cursor.execute(
        "SELECT scan_type, COUNT(*) as cnt FROM scan_history WHERE user_id=? GROUP BY scan_type",
        (user_id,)
    )
    by_type = {row['scan_type']: row['cnt'] for row in cursor.fetchall()}

    # By result
    cursor.execute(
        "SELECT result, COUNT(*) as cnt FROM scan_history WHERE user_id=? GROUP BY result",
        (user_id,)
    )
    by_result = {row['result']: row['cnt'] for row in cursor.fetchall()}

    # Recent 7 days activity
    cursor.execute(
        '''SELECT DATE(scanned_at) as day, COUNT(*) as cnt
           FROM scan_history
           WHERE user_id=? AND scanned_at >= datetime('now', '-7 days')
           GROUP BY DATE(scanned_at)
           ORDER BY day''',
        (user_id,)
    )
    daily = [{'day': r['day'], 'count': r['cnt']} for r in cursor.fetchall()]

    conn.close()

    return {
        'total_scans': total,
        'text_scans': by_type.get('text', 0),
        'image_scans': by_type.get('image', 0),
        'fake_count': by_result.get('Fake', 0),
        'real_count': by_result.get('Real', 0),
        'ai_generated_count': by_result.get('AI-Generated', 0),
        'manipulated_count': by_result.get('Manipulated', 0),
        'daily_activity': daily
    }


def get_scan_by_id(scan_id, user_id):
    """Get a single scan (with ownership check)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM scan_history WHERE id=? AND user_id=?",
        (scan_id, user_id)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


if __name__ == '__main__':
    init_db()
    print("Database setup complete.")
