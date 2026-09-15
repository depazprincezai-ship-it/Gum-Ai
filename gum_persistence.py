import os
import json
import time

try:
    import psycopg
except ImportError:
    psycopg = None

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
ENABLED = bool(DATABASE_URL and psycopg)


def _connect():
    if not ENABLED:
        raise RuntimeError("DATABASE_URL and psycopg are required for durable storage.")
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def init_store():
    if not ENABLED:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gum_users (
                    email TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gum_plugins (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    html_code TEXT NOT NULL,
                    session_id TEXT,
                    author_email TEXT,
                    author_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
        conn.commit()


def load_users():
    if not ENABLED:
        return None
    init_store()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT email, password_hash, display_name, created_at FROM gum_users")
            rows = cur.fetchall()
    return {
        row[0]: {
            "password_hash": row[1],
            "display_name": row[2],
            "created_at": row[3],
        }
        for row in rows
    }


def upsert_user(email, user):
    if not ENABLED:
        return
    init_store()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO gum_users(email, password_hash, display_name, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(email) DO UPDATE SET
                    password_hash = EXCLUDED.password_hash,
                    display_name = EXCLUDED.display_name,
                    created_at = EXCLUDED.created_at
            """, (
                email,
                user.get("password_hash", ""),
                user.get("display_name", ""),
                user.get("created_at", ""),
            ))
        conn.commit()


def load_plugins():
    if not ENABLED:
        return None
    init_store()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,name,html_code,session_id,author_email,author_name,created_at,updated_at FROM gum_plugins ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [
        {
            "id": r[0], "name": r[1], "html_code": r[2], "session_id": r[3],
            "author_email": r[4], "author_name": r[5], "created_at": r[6], "updated_at": r[7]
        }
        for r in rows
    ]


def save_plugins(plugins):
    if not ENABLED:
        return
    init_store()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM gum_plugins")
            for p in plugins:
                cur.execute("""
                    INSERT INTO gum_plugins(id,name,html_code,session_id,author_email,author_name,created_at,updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    p.get("id", ""), p.get("name", "Untitled plugin"), p.get("html_code", ""),
                    p.get("session_id"), p.get("author_email"), p.get("author_name", "Guest"),
                    p.get("created_at", ""), p.get("updated_at", ""),
                ))
        conn.commit()
