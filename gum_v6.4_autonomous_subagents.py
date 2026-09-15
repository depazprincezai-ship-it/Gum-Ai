from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import uuid
import secrets
from groq import Groq
import os
import re
import json
import base64
import datetime
import random
import math
import string
import hashlib
import platform
import shutil
import glob
import textwrap
import calendar
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import operator
import sqlite3
from functools import wraps

try:
    from openai import OpenAI
    _openai_sdk_available = True
except ImportError:
    _openai_sdk_available = False

app = Flask(__name__)

# ---------- Public deployment hardening ----------
# Set TRUST_PROXY_HEADERS=true only when Gum is behind a trusted reverse proxy
# (Cloudflare Tunnel, nginx, a hosting platform proxy, etc.).
TRUST_PROXY_HEADERS = os.environ.get("TRUST_PROXY_HEADERS", "false").lower() == "true"
if TRUST_PROXY_HEADERS:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "").strip()
if FLASK_SECRET_KEY:
    app.config["SECRET_KEY"] = FLASK_SECRET_KEY
else:
    # A random fallback keeps local development working, but public deployments
    # should always provide a persistent secret through .env / the host secret store.
    app.config["SECRET_KEY"] = secrets.token_hex(32)

PUBLIC_HOSTS = {h.strip().lower().split(":", 1)[0] for h in os.environ.get("GUM_PUBLIC_HOSTS", "").split(",") if h.strip()}
PUBLIC_HOSTS.update({"localhost", "127.0.0.1", "[::1]"})
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB hard cap on any single request body
app.config["JSON_SORT_KEYS"] = False
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.before_request
def public_deployment_guard():
    # Optional host allow-list. Leave GUM_PUBLIC_HOSTS empty for local-only
    # development; set it to your public hostname(s) when exposing Gum.
    if PUBLIC_HOSTS:
        host = request.host.lower().split(":", 1)[0]
        if host not in PUBLIC_HOSTS and host not in {"localhost", "127.0.0.1", "[::1]"}:
            return jsonify({"error": "Invalid host."}), 400

    # Public deployments must not expose Flask's interactive debugger.
    if request.path.startswith("/__debugger__"):
        return jsonify({"error": "Not found"}), 404


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    if request.is_secure:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return ensure_csrf_cookie(response)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (_openai_sdk_available and OPENAI_API_KEY) else None

# Providers available for AI Personas. Groq stays the default/free option;
# OpenAI only appears as selectable if OPENAI_API_KEY is set in .env AND
# the openai package is installed — otherwise it's silently omitted rather
# than shown as a broken option.
AI_PROVIDERS = {
    "groq": {
        "label": "Groq (free tier)",
        "available": True,
    },
    "openai": {
        "label": "OpenAI",
        "available": openai_client is not None,
    },
}

OPENAI_MODELS = {
    "gpt-4o-mini": "Fast, inexpensive, good default",
    "gpt-4o": "More capable, higher cost",
}


def call_ai_provider(provider, model, messages, temperature=1.0):
    """Unified chat-completion call across providers. Returns the reply text,
    or raises an exception the caller should catch and surface to the user."""
    if provider == "openai":
        if not openai_client:
            raise RuntimeError("OpenAI isn't configured on this server (missing OPENAI_API_KEY).")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response

DISPLAY_NAME = "Gum"
APP_VERSION = "6.1.0-mobile"

# ---------- Gum v6: public-service reliability layer ----------
PUBLIC_MODE = os.environ.get("GUM_PUBLIC_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
RATE_LIMIT_PER_MINUTE = max(10, int(os.environ.get("GUM_RATE_LIMIT_PER_MINUTE", "60")))
MAX_REQUEST_BYTES = max(1024 * 1024, int(os.environ.get("GUM_MAX_REQUEST_BYTES", str(8 * 1024 * 1024))))
_RATE_LOCK = threading.Lock()
_RATE_BUCKETS = {}
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

def _rate_key():
    try:
        identity = get_auth_from_request({})
    except Exception:
        identity = None
    return "user:" + identity if identity else "ip:" + (request.remote_addr or "unknown")

def _rate_allowed():
    now = time.time()
    cutoff = now - 60
    key = _rate_key()
    with _RATE_LOCK:
        hits = [stamp for stamp in _RATE_BUCKETS.get(key, []) if stamp >= cutoff]
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            _RATE_BUCKETS[key] = hits
            return False
        hits.append(now)
        _RATE_BUCKETS[key] = hits
        if len(_RATE_BUCKETS) > 5000:
            for old_key in list(_RATE_BUCKETS)[:1000]:
                _RATE_BUCKETS.pop(old_key, None)
        return True

@app.before_request
def gum_v6_rate_limit():
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _rate_allowed():
        return jsonify({"error": "Too many requests. Please try again shortly."}), 429

@app.errorhandler(413)
def gum_v6_too_large(_error):
    return jsonify({"error": "Request is too large."}), 413

@app.route("/ready", methods=["GET"])
def gum_ready():
    return jsonify({"ok": True, "service": "gum", "version": APP_VERSION})

@app.route("/version", methods=["GET"])
def gum_version():
    return jsonify({"service": "gum", "version": APP_VERSION, "public_mode": PUBLIC_MODE})

CURRENT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_HISTORY = 20
MAX_CONTEXT_CHARS = int(os.environ.get("GUM_MAX_CONTEXT_CHARS", "24000"))
NOTES_FILE = "notes.json"
CHATS_DIR = "saved_chats"
os.makedirs(CHATS_DIR, exist_ok=True)
MEMORY_DB = os.environ.get("MEMORY_DB", "gum_memory.db")
MEMORY_MAX_CHATS = int(os.environ.get("GUM_MEMORY_MAX_CHATS", "100"))
MEMORY_MAX_MESSAGES_PER_CHAT = int(os.environ.get("GUM_MEMORY_MAX_MESSAGES", "200"))

GENERATED_DIR = "static/generated"
UPLOADED_DIR = "static/uploaded"
FILES_DIR = "static/files"
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

ALLOWED_FILE_EXTENSIONS = {"txt", "md", "csv", "json", "py", "js", "html", "css", "log", "pdf"}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB, keeps things safe on Termux/free hosting

# ---------- Accounts (sign up / sign in) ----------
# The owner is whoever's email matches OWNER_EMAIL in your .env — never hardcoded in code.
# Set it in .env as: OWNER_EMAIL=youraddress@example.com
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "").strip().lower()

USERS_FILE = "users.json"
SESSION_DB = os.environ.get("SESSION_DB", "gum_sessions.db")
SESSION_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
AUTH_COOKIE_NAME = "gum_session"
AUTH_COOKIE_MAX_AGE = SESSION_TOKEN_TTL_SECONDS
MIN_PASSWORD_LENGTH = 8
ALLOW_LEGACY_AUTH_TOKENS = os.environ.get("ALLOW_LEGACY_AUTH_TOKENS", "0").strip().lower() in {"1", "true", "yes", "on"}
CSRF_COOKIE_NAME = "gum_csrf"
CSRF_HEADER_NAME = "X-Gum-CSRF"

# Browser sessions use an HttpOnly, SameSite cookie. Session records are persisted
# in SQLite so a restart does not silently invalidate every browser session.
# Only a SHA-256 hash of the bearer token is stored on disk.
# The legacy in-memory map is retained only for tokens issued by older builds.


def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


users_store = load_users()          # email -> {password_hash, display_name, created_at}
active_sessions = {}                # legacy token -> {email, expires_at}


def _session_db_connect():
    conn = sqlite3.connect(SESSION_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_session_store():
    conn = _session_db_connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                revoked_at REAL,
                session_id TEXT,
                user_agent TEXT,
                ip_hash TEXT,
                last_seen REAL
            )
        """)
        # Upgrade databases created by Phase 1.2/1.3 without downtime.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(auth_sessions)").fetchall()}
        if "session_id" not in columns:
            conn.execute("ALTER TABLE auth_sessions ADD COLUMN session_id TEXT")
        if "user_agent" not in columns:
            conn.execute("ALTER TABLE auth_sessions ADD COLUMN user_agent TEXT")
        if "ip_hash" not in columns:
            conn.execute("ALTER TABLE auth_sessions ADD COLUMN ip_hash TEXT")
        if "last_seen" not in columns:
            conn.execute("ALTER TABLE auth_sessions ADD COLUMN last_seen REAL")
        rows = conn.execute("SELECT token_hash FROM auth_sessions WHERE session_id IS NULL OR session_id = ''").fetchall()
        for row in rows:
            conn.execute("UPDATE auth_sessions SET session_id = ? WHERE token_hash = ?", (secrets.token_urlsafe(18), row[0]))
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_sessions_session_id ON auth_sessions(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_email ON auth_sessions(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at)")
        conn.commit()
    finally:
        conn.close()


def _hash_session_token(token):
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def cleanup_expired_sessions():
    now = time.time()
    conn = _session_db_connect()
    try:
        conn.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL",
            (now,),
        )
        conn.commit()
    finally:
        conn.close()


def is_valid_email(email):
    return bool(re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email or ""))


def create_user(email, password, display_name):
    email = email.strip().lower()
    if email in users_store:
        return None, "An account with that email already exists."
    if not is_valid_email(email):
        return None, "That doesn't look like a valid email."
    if len(password) < MIN_PASSWORD_LENGTH:
        return None, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."

    users_store[email] = {
        "password_hash": generate_password_hash(password),
        "display_name": (display_name or email.split("@")[0])[:40],
        "created_at": datetime.datetime.now().isoformat(),
    }
    save_users(users_store)
    return users_store[email], None


def verify_login(email, password):
    email = email.strip().lower()
    user = users_store.get(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return False
    return True


def _request_ip_hash():
    ip = get_client_ip() if request else "unknown"
    return hashlib.sha256((ip + app.config["SECRET_KEY"]).encode("utf-8")).hexdigest()[:24]


def create_session_token(email):
    token = secrets.token_hex(32)
    now = time.time()
    expires_at = now + SESSION_TOKEN_TTL_SECONDS
    normalized_email = email.strip().lower()
    token_hash = _hash_session_token(token)
    session_id = secrets.token_urlsafe(18)
    user_agent = (request.headers.get("User-Agent", "") or "")[:240]
    ip_hash = _request_ip_hash()
    conn = _session_db_connect()
    try:
        conn.execute(
            "INSERT INTO auth_sessions (token_hash, email, created_at, expires_at, session_id, user_agent, ip_hash, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (token_hash, normalized_email, now, expires_at, session_id, user_agent, ip_hash, now),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def get_user_from_token(token):
    if not token:
        return None
    token_hash = _hash_session_token(token)
    now = time.time()
    conn = _session_db_connect()
    try:
        row = conn.execute(
            "SELECT email, expires_at, revoked_at FROM auth_sessions WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
    finally:
        conn.close()

    if row:
        if row["revoked_at"] is not None or now >= row["expires_at"]:
            revoke_session_token(token)
            return None
        # Touch activity without extending the absolute 30-day expiry.
        if now - (row["expires_at"] - SESSION_TOKEN_TTL_SECONDS) >= 60:
            conn2 = _session_db_connect()
            try:
                conn2.execute("UPDATE auth_sessions SET last_seen = ? WHERE token_hash = ?", (now, token_hash))
                conn2.commit()
            finally:
                conn2.close()
        return row["email"]

    # Compatibility for pre-phase persistent-session tokens.
    session = active_sessions.get(token)
    if not session:
        return None
    if now >= session["expires_at"]:
        active_sessions.pop(token, None)
        return None
    return session["email"]


def revoke_session_token(token):
    if not token:
        return
    active_sessions.pop(token, None)
    token_hash = _hash_session_token(token)
    conn = _session_db_connect()
    try:
        conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (time.time(), token_hash),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions(email, current_token=""):
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return []
    conn = _session_db_connect()
    try:
        rows = conn.execute(
            "SELECT session_id, token_hash, created_at, expires_at, revoked_at, user_agent, ip_hash, last_seen FROM auth_sessions "
            "WHERE email = ? AND revoked_at IS NULL AND expires_at > ? ORDER BY created_at DESC",
            (normalized_email, time.time()),
        ).fetchall()
    finally:
        conn.close()

    current_hash = _hash_session_token(current_token) if current_token else ""
    return [
        {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "revoked": row["revoked_at"] is not None,
            "current": bool(current_hash and row["token_hash"] == current_hash),
            "device": (row["user_agent"] or "Unknown browser")[:120],
            "network": (row["ip_hash"] or "")[:12],
            "last_seen": row["last_seen"] or row["created_at"],
        }
        for row in rows
    ]


def revoke_session_by_id(email, session_id):
    normalized_email = (email or "").strip().lower()
    session_id = str(session_id or "").strip()
    if not normalized_email or not session_id or len(session_id) > 80:
        return False
    conn = _session_db_connect()
    try:
        cur = conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE session_id = ? AND email = ? AND revoked_at IS NULL",
            (time.time(), session_id, normalized_email),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def revoke_all_sessions(email):
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return
    conn = _session_db_connect()
    try:
        conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE email = ? AND revoked_at IS NULL",
            (time.time(), normalized_email),
        )
        conn.commit()
    finally:
        conn.close()
    for token, session in list(active_sessions.items()):
        if session.get("email") == normalized_email:
            active_sessions.pop(token, None)


init_session_store()

# ---------- AI telemetry / usage store (Phase 3) ----------
USAGE_DB = os.environ.get("USAGE_DB", "gum_usage.db")

def _usage_db_connect():
    conn = sqlite3.connect(USAGE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_usage_store():
    conn = _usage_db_connect()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS ai_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            email TEXT NOT NULL,
            session_id TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            route TEXT NOT NULL,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            success INTEGER NOT NULL DEFAULT 1,
            error_type TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_usage_email_time ON ai_usage(email, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_usage_model_time ON ai_usage(model, created_at)")
        conn.commit()
    finally:
        conn.close()

def record_ai_usage(email, session_id, provider, model, route, latency_ms, usage=None, success=True, error_type=None):
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    conn = _usage_db_connect()
    try:
        conn.execute("""INSERT INTO ai_usage
            (created_at,email,session_id,provider,model,route,latency_ms,prompt_tokens,completion_tokens,total_tokens,success,error_type)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), (email or "").strip().lower(), session_id, provider, model, route,
             max(0, int(latency_ms)), prompt_tokens, completion_tokens, total_tokens,
             1 if success else 0, error_type))
        conn.commit()
    finally:
        conn.close()

def usage_summary(email, days=30):
    since = time.time() - max(1, min(int(days), 365)) * 86400
    conn = _usage_db_connect()
    try:
        rows = conn.execute("""SELECT provider, model, route, COUNT(*) AS requests,
            SUM(success) AS successful, SUM(total_tokens) AS tokens, AVG(latency_ms) AS avg_latency_ms
            FROM ai_usage WHERE email=? AND created_at>=? GROUP BY provider, model, route
            ORDER BY requests DESC""", ((email or "").strip().lower(), since)).fetchall()
        totals = conn.execute("""SELECT COUNT(*) AS requests, SUM(success) AS successful,
            SUM(total_tokens) AS tokens, AVG(latency_ms) AS avg_latency_ms
            FROM ai_usage WHERE email=? AND created_at>=?""", ((email or "").strip().lower(), since)).fetchone()
    finally:
        conn.close()
    return {
        "days": max(1, min(int(days), 365)),
        "totals": {
            "requests": int(totals["requests"] or 0),
            "successful": int(totals["successful"] or 0),
            "tokens": int(totals["tokens"] or 0),
            "avg_latency_ms": round(float(totals["avg_latency_ms"] or 0), 1),
        },
        "breakdown": [dict(r) for r in rows],
    }

init_usage_store()
cleanup_expired_sessions()

def is_owner(email):
    return bool(email) and bool(OWNER_EMAIL) and email.strip().lower() == OWNER_EMAIL


def get_auth_from_request(data=None):
    """Resolve the authenticated user from the secure cookie first, then the legacy body token."""
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    email = get_user_from_token(cookie_token)
    if email:
        return email
    if ALLOW_LEGACY_AUTH_TOKENS:
        token = data.get("auth_token") if data else None
        return get_user_from_token(token)
    return None


def set_auth_cookie(response, token):
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        max_age=AUTH_COOKIE_MAX_AGE,
        httponly=True,
        secure=request.is_secure,
        samesite="Lax",
        path="/",
    )
    return response


def clear_auth_cookie(response):
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return response


def ensure_csrf_cookie(response):
    token = request.cookies.get(CSRF_COOKIE_NAME) or secrets.token_urlsafe(24)
    if not request.cookies.get(CSRF_COOKIE_NAME):
        response.set_cookie(CSRF_COOKIE_NAME, token, max_age=AUTH_COOKIE_MAX_AGE, httponly=False, secure=request.is_secure, samesite="Lax", path="/")
    return response


def csrf_valid_for_cookie_auth():
    if request.method in SAFE_METHODS or not request.cookies.get(AUTH_COOKIE_NAME):
        return True
    cookie = request.cookies.get(CSRF_COOKIE_NAME, "")
    header = request.headers.get(CSRF_HEADER_NAME, "")
    return bool(cookie and header and secrets.compare_digest(cookie, header))


def auth_required(data=None):
    """Return (email, None) when authenticated, otherwise (None, JSON response)."""
    email = get_auth_from_request(data)
    if not email or email not in users_store:
        return None, (jsonify({"error": "Sign in required."}), 401)
    return email, None


def scoped_session_id(session_id, email):
    """Keep in-memory conversation keys isolated between authenticated users."""
    raw = str(session_id or "default")[:120]
    if email:
        owner = hashlib.sha256(email.encode("utf-8")).hexdigest()[:24]
        return f"user:{owner}:{raw}"
    return f"guest:{raw}"


# ---------- Request integrity ----------
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
AUTH_EXEMPT_ENDPOINTS = {"signup", "login", "unlock_site", "access_status", "static"}


@app.before_request
def enforce_same_origin_for_cookie_auth():
    if request.method in SAFE_METHODS or request.endpoint in AUTH_EXEMPT_ENDPOINTS:
        return None
    if not request.cookies.get(AUTH_COOKIE_NAME):
        return None
    origin = request.headers.get("Origin")
    if origin:
        expected = f"{request.scheme}://{request.host}"
        if origin.rstrip("/") != expected.rstrip("/"):
            return jsonify({"error": "Cross-site request blocked."}), 403
    if not csrf_valid_for_cookie_auth():
        return jsonify({"error": "Request integrity check failed. Refresh the page and try again."}), 403
    return None


# ---------- Access control & rate limiting ----------
# Optional password gate. Leave ACCESS_PASSWORD unset in .env to allow open access.
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "")

# Per-IP request limits (in-memory, resets on server restart)
RATE_LIMIT_WINDOW_SECONDS = 3600  # 1 hour window
RATE_LIMIT_MAX_REQUESTS = 40      # max chat/generate/image calls per IP per window
RATE_LIMIT_MAX_GENERATIONS = 10   # separate, stricter cap for image generation (heavier cost)

request_log = {}      # ip -> list of timestamps (chat/general)
generation_log = {}   # ip -> list of timestamps (image generation)
unlocked_ips = set()  # IPs that entered the correct access password this session


def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit(ip, log_dict, max_requests, window=RATE_LIMIT_WINDOW_SECONDS):
    now = time.time()
    timestamps = log_dict.get(ip, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= max_requests:
        log_dict[ip] = timestamps
        return False
    timestamps.append(now)
    log_dict[ip] = timestamps
    return True


def is_unlocked(ip):
    return not ACCESS_PASSWORD or ip in unlocked_ips

# ---------- Model catalog (fetched live from Groq, so it never goes stale) ----------
FALLBACK_MODELS = {
    "openai/gpt-oss-120b": "Most capable, general use",
    "openai/gpt-oss-20b": "Fast, lighter, good default",
    "groq/compound": "Web search + code execution built in",
    "groq/compound-mini": "Lighter agentic system",
}

# Models that appear in Groq's /models list but aren't safe/sensible as general chat picks
# (audio/transcription models, moderation-only models, deprecated/enterprise-only text models)
MODEL_EXCLUDE_PATTERNS = ("whisper", "guard", "prompt-guard", "tts")

_model_cache = {"models": None, "fetched_at": 0}
MODEL_CACHE_TTL_SECONDS = 3600  # refresh the live list at most once an hour


def fetch_live_models():
    """Pulls the real, current model list from Groq's API. Falls back to a
    small hardcoded set if the call fails, so the app never breaks over this."""
    now = time.time()
    if _model_cache["models"] and (now - _model_cache["fetched_at"] < MODEL_CACHE_TTL_SECONDS):
        return _model_cache["models"]

    try:
        response = client.models.list()
        models = {}
        for m in response.data:
            model_id = m.id
            if any(bad in model_id.lower() for bad in MODEL_EXCLUDE_PATTERNS):
                continue
            models[model_id] = "Available on Groq"
        # keep our known-good descriptions where we have them
        for known_id, desc in FALLBACK_MODELS.items():
            if known_id in models:
                models[known_id] = desc
        if models:
            _model_cache["models"] = models
            _model_cache["fetched_at"] = now
            return models
    except Exception:
        pass

    return _model_cache["models"] or FALLBACK_MODELS


AVAILABLE_MODELS = fetch_live_models()

DEFAULT_SYSTEM_PROMPT = f"""Your name is {DISPLAY_NAME}. You can refer to yourself as {DISPLAY_NAME} when natural.

You are a highly capable, thoughtful AI assistant. Think carefully before answering, prioritize
accuracy over speed, and be honest about uncertainty rather than guessing. Be clear and direct,
avoid unnecessary filler, and match your tone to the conversation. When writing code, prioritize
correctness and readability. If a request is ambiguous, make a reasonable assumption and state it."""

# per-session state
conversations = {}
notes_store = []

# per-session settings: model, temperature, system prompt, font size, theme
session_settings = {}

DEFAULT_SETTINGS = {
    "model": CURRENT_MODEL,
    "temperature": 1.0,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "theme": "dusk",           # dusk | forest | mono
    "font_size": "medium",     # small | medium | large
    "autoscroll": True,
    "background_url": None,    # gallery image URL used as chat background
    "text_color": None,        # hex color for message text, from the color wheel
    "bubble_opacity": 0.85,    # 0.3 - 1.0, keeps background visible behind bubbles
    "response_mode": "thinking",  # thinking | instant — affects how carefully Gum answers
}

# Response mode presets — "thinking" favors accuracy and reasoning depth,
# "instant" favors speed with a lighter, more direct answer style.
RESPONSE_MODE_PRESETS = {
    "thinking": {
        "temperature_override": None,   # keep whatever the user set on the Creativity slider
        "extra_instruction": (
            "\n\nTake a moment to think through the problem step by step before answering. "
            "Prioritize accuracy and completeness over speed."
        ),
    },
    "instant": {
        "temperature_override": None,
        "extra_instruction": (
            "\n\nAnswer as directly and quickly as possible. Skip step-by-step reasoning out loud — "
            "give the shortest correct answer that fully addresses the question."
        ),
    },
}


def get_settings(session_id):
    if session_id not in session_settings:
        session_settings[session_id] = DEFAULT_SETTINGS.copy()
    return session_settings[session_id]


def build_system_message(settings):
    """Combines the user's custom system prompt with the active response-mode
    instruction (Thinking vs Instant), so every chat call gets consistent behavior."""
    base_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    mode = settings.get("response_mode", "thinking")
    preset = RESPONSE_MODE_PRESETS.get(mode, RESPONSE_MODE_PRESETS["thinking"])
    return {"role": "system", "content": base_prompt + preset["extra_instruction"]}


def load_notes():
    global notes_store
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r") as f:
                notes_store = json.load(f)
        except (json.JSONDecodeError, IOError):
            notes_store = []
    return notes_store


def save_notes():
    with open(NOTES_FILE, "w") as f:
        json.dump(notes_store, f, indent=2)


load_notes()


# ---------- Persistent memory / chat store (Phase 4) ----------
def _memory_db_connect():
    conn = sqlite3.connect(MEMORY_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_memory_store():
    conn = _memory_db_connect()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_email TEXT NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT 'New chat',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(owner_email, session_id)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            seq INTEGER NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(chat_id, seq)
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_owner_updated ON chats(owner_email, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_seq ON messages(chat_id, seq)")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    finally:
        conn.close()

init_memory_store()

def _normalize_chat_title(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return (text[:40] + "…") if len(text) > 40 else (text or "New chat")

def persist_chat(owner_email, session_id, history, title=None):
    if not owner_email or not session_id or not history:
        return
    email = owner_email.strip().lower()
    now = time.time()
    first_user = next((m.get("content", "") for m in history if m.get("role") == "user"), "New chat")
    title = _normalize_chat_title(title or first_user)
    messages = history[-MEMORY_MAX_MESSAGES_PER_CHAT:]
    conn = _memory_db_connect()
    try:
        conn.execute("""INSERT INTO chats(owner_email,session_id,title,created_at,updated_at)
            VALUES(?,?,?,?,?) ON CONFLICT(owner_email,session_id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at""",
            (email, str(session_id), title, now, now))
        row = conn.execute("SELECT id FROM chats WHERE owner_email=? AND session_id=?", (email, str(session_id))).fetchone()
        chat_id = row["id"]
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        for seq, msg in enumerate(messages):
            role = msg.get("role", "user")
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(msg.get("content", ""))
            if content:
                conn.execute("INSERT INTO messages(chat_id,role,content,seq,created_at) VALUES(?,?,?,?,?)",
                             (chat_id, role, content, seq, now))
        # Keep the newest N chats for each account.
        old = conn.execute("SELECT id FROM chats WHERE owner_email=? ORDER BY updated_at DESC LIMIT -1 OFFSET ?",
                           (email, max(1, MEMORY_MAX_CHATS))).fetchall()
        for row in old:
            conn.execute("DELETE FROM chats WHERE id=?", (row["id"],))
        conn.commit()
    finally:
        conn.close()

def load_memory_chat(owner_email, session_id):
    if not owner_email or not session_id:
        return None
    conn = _memory_db_connect()
    try:
        chat = conn.execute("SELECT id,session_id,title,created_at,updated_at FROM chats WHERE owner_email=? AND session_id=?",
                            (owner_email.strip().lower(), str(session_id))).fetchone()
        if not chat:
            return None
        rows = conn.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY seq", (chat["id"],)).fetchall()
        return {"session_id": chat["session_id"], "title": chat["title"], "updated_at": datetime.datetime.fromtimestamp(chat["updated_at"]).isoformat(),
                "owner_email": owner_email.strip().lower(), "messages": [{"role": r["role"], "content": r["content"]} for r in rows]}
    finally:
        conn.close()

def list_memory_chats(owner_email, query=""):
    email = (owner_email or "").strip().lower()
    q = (query or "").strip()
    conn = _memory_db_connect()
    try:
        if q:
            rows = conn.execute("""SELECT DISTINCT c.session_id,c.title,c.updated_at FROM chats c
                LEFT JOIN messages m ON m.chat_id=c.id
                WHERE c.owner_email=? AND (c.title LIKE ? OR m.content LIKE ?)
                ORDER BY c.updated_at DESC LIMIT 100""", (email, f"%{q}%", f"%{q}%")).fetchall()
        else:
            rows = conn.execute("SELECT session_id,title,updated_at FROM chats WHERE owner_email=? ORDER BY updated_at DESC LIMIT 100", (email,)).fetchall()
        return [{"session_id": r["session_id"], "title": r["title"], "updated_at": datetime.datetime.fromtimestamp(r["updated_at"]).isoformat()} for r in rows]
    finally:
        conn.close()

def rename_memory_chat(owner_email, session_id, title):
    title = _normalize_chat_title(title)
    conn = _memory_db_connect()
    try:
        cur = conn.execute("UPDATE chats SET title=?, updated_at=? WHERE owner_email=? AND session_id=?",
                           (title, time.time(), (owner_email or "").strip().lower(), str(session_id)))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

def delete_memory_chat(owner_email, session_id):
    conn = _memory_db_connect()
    try:
        cur = conn.execute("DELETE FROM chats WHERE owner_email=? AND session_id=?", ((owner_email or "").strip().lower(), str(session_id)))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

# ---------- Auto-save chats ----------
def chat_owner_key(owner_email):
    if not owner_email:
        return None
    return hashlib.sha256(owner_email.strip().lower().encode("utf-8")).hexdigest()[:24]


def chat_file_path(session_id, owner_email=None):
    owner_key = chat_owner_key(owner_email)
    if not owner_key:
        return None
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', str(session_id or "default"))[:120]
    return os.path.join(CHATS_DIR, f"{owner_key}_{safe_id}.json")


def autosave_chat(session_id, history, owner_email=None):
    if not history or not owner_email:
        return
    first_user_msg = next((m["content"] for m in history if m["role"] == "user"), "New chat")
    title = (first_user_msg[:40] + "…") if len(first_user_msg) > 40 else first_user_msg
    payload = {
        "session_id": session_id,
        "title": title,
        "updated_at": datetime.datetime.now().isoformat(),
        "owner_email": owner_email,
        "messages": history,
    }
    try:
        persist_chat(owner_email, session_id, history, title)
    except Exception:
        pass
    try:
        with open(chat_file_path(session_id, owner_email), "w") as f:
            json.dump(payload, f, indent=2)
    except IOError:
        pass


def list_saved_chats(owner_email):
    chats = []
    owner_key = chat_owner_key(owner_email)
    if not owner_key:
        return chats
    prefix = f"{owner_key}_"
    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json") or not fname.startswith(prefix):
            continue
        try:
            with open(os.path.join(CHATS_DIR, fname), "r") as f:
                data = json.load(f)
            if data.get("owner_email", "").strip().lower() != owner_email.strip().lower():
                continue
            chats.append({
                "session_id": data.get("session_id", "default"),
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
            })
        except (IOError, json.JSONDecodeError):
            continue
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


def load_saved_chat(session_id, owner_email):
    path = chat_file_path(session_id, owner_email)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("owner_email", "").strip().lower() != owner_email.strip().lower():
            return None
        return data
    except (IOError, json.JSONDecodeError):
        return None


def delete_saved_chat(session_id, owner_email):
    path = chat_file_path(session_id, owner_email)
    if path and os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------- Plugin storage ----------
PLUGINS_FILE = "plugins.json"
MAX_PLUGIN_CODE_SIZE = 100_000  # ~100KB per plugin, plenty for a hobby script


def load_plugins():
    if os.path.exists(PLUGINS_FILE):
        try:
            with open(PLUGINS_FILE, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return []
    return []


def save_plugins(plugins):
    with open(PLUGINS_FILE, "w") as f:
        json.dump(plugins, f, indent=2)


plugins_store = load_plugins()


# ---------- Built-in (permanent) plugins ----------
# These ship with the app, are always available to everyone, and can't be
# edited or deleted by users (only the account "author_email": None + is_builtin
# flag matters — the delete/edit routes check for is_builtin before anything else).
BUILTIN_PLUGINS = [
    {
        "id": "builtin-qr-generator",
        "name": "QR Code Generator",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{font-family:sans-serif;background:#1a1918;color:#f2ecdc;padding:20px;margin:0;}
input{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #444;background:#242220;color:#fff;font-size:14px;margin-bottom:12px;}
button{width:100%;padding:10px;border-radius:8px;border:none;background:#e39a5c;color:#1a1918;font-weight:600;cursor:pointer;}
#qr{margin-top:16px;text-align:center;}
#qr img{max-width:100%;border-radius:8px;background:#fff;padding:10px;}
</style></head><body>
<h3>QR Code Generator</h3>
<input id="text" placeholder="Enter text or a URL">
<button onclick="makeQR()">Generate</button>
<div id="qr"></div>
<script>
function makeQR() {
  const text = document.getElementById('text').value.trim();
  if (!text) return;
  const url = 'https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=' + encodeURIComponent(text);
  document.getElementById('qr').innerHTML = '<img src="' + url + '" alt="QR code">';
}
</script>
</body></html>""",
    },
    {
        "id": "builtin-unit-converter",
        "name": "Unit Converter",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{font-family:sans-serif;background:#1a1918;color:#f2ecdc;padding:20px;margin:0;}
select,input{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #444;background:#242220;color:#fff;font-size:14px;margin-bottom:10px;}
#result{margin-top:14px;font-size:20px;font-weight:600;color:#e39a5c;}
label{font-size:12px;opacity:0.7;}
</style></head><body>
<h3>Unit Converter</h3>
<label>Category</label>
<select id="category" onchange="updateUnits()">
  <option value="length">Length</option>
  <option value="weight">Weight</option>
  <option value="temp">Temperature</option>
</select>
<label>From</label>
<select id="from"></select>
<label>To</label>
<select id="to"></select>
<input id="value" type="number" placeholder="Enter value" oninput="convert()">
<div id="result"></div>
<script>
const units = {
  length: { m: 1, km: 1000, cm: 0.01, mi: 1609.34, ft: 0.3048, in: 0.0254 },
  weight: { kg: 1, g: 0.001, lb: 0.453592, oz: 0.0283495 },
  temp: { c: 'c', f: 'f', k: 'k' }
};
function updateUnits() {
  const cat = document.getElementById('category').value;
  const from = document.getElementById('from');
  const to = document.getElementById('to');
  from.innerHTML = ''; to.innerHTML = '';
  Object.keys(units[cat]).forEach(u => {
    from.innerHTML += `<option value="${u}">${u}</option>`;
    to.innerHTML += `<option value="${u}">${u}</option>`;
  });
  convert();
}
function convert() {
  const cat = document.getElementById('category').value;
  const from = document.getElementById('from').value;
  const to = document.getElementById('to').value;
  const val = parseFloat(document.getElementById('value').value) || 0;
  let result;
  if (cat === 'temp') {
    let celsius = from === 'c' ? val : from === 'f' ? (val - 32) * 5/9 : val - 273.15;
    result = to === 'c' ? celsius : to === 'f' ? celsius * 9/5 + 32 : celsius + 273.15;
  } else {
    result = val * units[cat][from] / units[cat][to];
  }
  document.getElementById('result').textContent = `${val} ${from} = ${result.toFixed(4)} ${to}`;
}
updateUnits();
document.getElementById('from').addEventListener('change', convert);
document.getElementById('to').addEventListener('change', convert);
</script>
</body></html>""",
    },
    {
        "id": "builtin-json-formatter",
        "name": "JSON Formatter",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{font-family:monospace;background:#1a1918;color:#f2ecdc;padding:16px;margin:0;}
textarea{width:100%;box-sizing:border-box;height:140px;padding:10px;border-radius:8px;border:1px solid #444;background:#242220;color:#fff;font-family:monospace;font-size:13px;}
button{padding:9px 16px;border-radius:8px;border:none;background:#e39a5c;color:#1a1918;font-weight:600;cursor:pointer;margin:10px 4px 10px 0;}
pre{background:#0f0e0d;padding:12px;border-radius:8px;overflow-x:auto;font-size:12px;white-space:pre-wrap;word-break:break-all;}
.error{color:#d97757;}
</style></head><body>
<h3 style="font-family:sans-serif;">JSON Formatter</h3>
<textarea id="input" placeholder='Paste JSON here, e.g. {"a":1,"b":[2,3]}'></textarea><br>
<button onclick="format()">Format</button>
<button onclick="minify()">Minify</button>
<pre id="output"></pre>
<script>
function format() {
  try {
    const parsed = JSON.parse(document.getElementById('input').value);
    document.getElementById('output').textContent = JSON.stringify(parsed, null, 2);
    document.getElementById('output').className = '';
  } catch (e) {
    document.getElementById('output').textContent = 'Invalid JSON: ' + e.message;
    document.getElementById('output').className = 'error';
  }
}
function minify() {
  try {
    const parsed = JSON.parse(document.getElementById('input').value);
    document.getElementById('output').textContent = JSON.stringify(parsed);
    document.getElementById('output').className = '';
  } catch (e) {
    document.getElementById('output').textContent = 'Invalid JSON: ' + e.message;
    document.getElementById('output').className = 'error';
  }
}
</script>
</body></html>""",
    },
    {
        "id": "builtin-color-palette",
        "name": "Color Palette Generator",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{font-family:sans-serif;background:#1a1918;color:#f2ecdc;padding:16px;margin:0;}
button{padding:10px 16px;border-radius:8px;border:none;background:#e39a5c;color:#1a1918;font-weight:600;cursor:pointer;margin-bottom:14px;}
.palette{display:flex;gap:8px;flex-wrap:wrap;}
.swatch{width:70px;height:70px;border-radius:10px;display:flex;align-items:flex-end;justify-content:center;padding-bottom:6px;font-size:10px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.3);}
</style></head><body>
<h3>Color Palette Generator</h3>
<button onclick="generate()">🎲 Generate new palette</button>
<div class="palette" id="palette"></div>
<script>
function randomColor() {
  const hue = Math.floor(Math.random() * 360);
  const sat = 55 + Math.floor(Math.random() * 30);
  const light = 45 + Math.floor(Math.random() * 25);
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}
function hslToHex(h, s, l) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = `hsl(${h},${s}%,${l}%)`;
  return ctx.fillStyle;
}
function generate() {
  const container = document.getElementById('palette');
  container.innerHTML = '';
  for (let i = 0; i < 6; i++) {
    const color = randomColor();
    const swatch = document.createElement('div');
    swatch.className = 'swatch';
    swatch.style.background = color;
    swatch.style.color = '#000';
    swatch.textContent = 'copy';
    swatch.addEventListener('click', () => {
      navigator.clipboard?.writeText(color);
      swatch.textContent = 'copied!';
      setTimeout(() => swatch.textContent = 'copy', 800);
    });
    container.appendChild(swatch);
  }
}
generate();
</script>
</body></html>""",
    },
]


def add_plugin(name, html_code, session_id, author_email, author_name):
    entry = {
        "id": uuid.uuid4().hex,
        "name": name.strip()[:60] or "Untitled plugin",
        "html_code": html_code,
        "session_id": session_id,
        "author_email": author_email,       # None for anonymous/guest authors
        "author_name": author_name or "Guest",
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    plugins_store.insert(0, entry)
    save_plugins(plugins_store)
    return entry


def update_plugin(plugin_id, name, html_code):
    for p in plugins_store:
        if p["id"] == plugin_id:
            p["name"] = name.strip()[:60] or "Untitled plugin"
            p["html_code"] = html_code
            p["updated_at"] = datetime.datetime.now().isoformat()
            save_plugins(plugins_store)
            return p
    return None


def find_plugin(plugin_id):
    for p in BUILTIN_PLUGINS:
        if p["id"] == plugin_id:
            return p
    for p in plugins_store:
        if p["id"] == plugin_id:
            return p
    return None


def get_all_plugins():
    """Built-ins first (they're permanent fixtures), then user plugins newest-first."""
    return BUILTIN_PLUGINS + plugins_store


def delete_plugin(plugin_id):
    global plugins_store
    before = len(plugins_store)
    plugins_store = [p for p in plugins_store if p["id"] != plugin_id]
    save_plugins(plugins_store)
    return len(plugins_store) < before


# ---------- AI Personas ----------
# A persona is a configured character — name, avatar, system prompt, provider,
# and model — layered on top of real providers (Groq / OpenAI). It is not a
# new trained model; it's the same "custom bot" idea as OpenAI's GPTs.
PERSONAS_FILE = "personas.json"
MAX_PERSONA_PROMPT_SIZE = 6000


def load_personas():
    if os.path.exists(PERSONAS_FILE):
        try:
            with open(PERSONAS_FILE, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return []
    return []


def save_personas(personas):
    with open(PERSONAS_FILE, "w") as f:
        json.dump(personas, f, indent=2)


personas_store = load_personas()


def add_persona(name, avatar, system_prompt, provider, model, author_email, author_name):
    entry = {
        "id": uuid.uuid4().hex,
        "name": name.strip()[:40] or "Untitled Persona",
        "avatar": (avatar or "🤖")[:4],
        "system_prompt": system_prompt.strip()[:MAX_PERSONA_PROMPT_SIZE],
        "provider": provider if provider in AI_PROVIDERS else "groq",
        "model": model,
        "author_email": author_email,
        "author_name": author_name or "Guest",
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    personas_store.insert(0, entry)
    save_personas(personas_store)
    return entry


def find_persona(persona_id):
    for p in personas_store:
        if p["id"] == persona_id:
            return p
    return None


def update_persona(persona_id, name, avatar, system_prompt, provider, model):
    for p in personas_store:
        if p["id"] == persona_id:
            p["name"] = name.strip()[:40] or "Untitled Persona"
            p["avatar"] = (avatar or "🤖")[:4]
            p["system_prompt"] = system_prompt.strip()[:MAX_PERSONA_PROMPT_SIZE]
            p["provider"] = provider if provider in AI_PROVIDERS else "groq"
            p["model"] = model
            p["updated_at"] = datetime.datetime.now().isoformat()
            save_personas(personas_store)
            return p
    return None


def delete_persona(persona_id):
    global personas_store
    before = len(personas_store)
    personas_store = [p for p in personas_store if p["id"] != persona_id]
    save_personas(personas_store)
    return len(personas_store) < before


# ---------- Image storage ----------
IMAGE_INDEX_FILE = "image_index.json"


def load_image_index():
    if os.path.exists(IMAGE_INDEX_FILE):
        try:
            with open(IMAGE_INDEX_FILE, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return []
    return []


def save_image_index(index):
    with open(IMAGE_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)


image_index = load_image_index()


# ---------- Private file ownership index ----------
FILE_INDEX_FILE = "file_index.json"

def load_file_index():
    if os.path.exists(FILE_INDEX_FILE):
        try:
            with open(FILE_INDEX_FILE, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}
    return {}

def save_file_index(index):
    with open(FILE_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

file_index = load_file_index()

def register_private_file(filename, owner_email, session_id=None):
    if not filename or not owner_email:
        return
    file_index[filename] = {
        "owner_email": owner_email.strip().lower(),
        "session_id": session_id,
        "created_at": datetime.datetime.now().isoformat(),
    }
    save_file_index(file_index)

def private_file_allowed(filename, owner_email):
    meta = file_index.get(filename)
    return bool(meta and owner_email and meta.get("owner_email") == owner_email.strip().lower())


def add_to_image_index(entry):
    image_index.insert(0, entry)  # newest first
    save_image_index(image_index)


def generate_image(prompt, session_id, owner_email=None):
    """Calls Pollinations.ai's current unified endpoint and saves the result locally.
    Falls back to the legacy endpoint if the new one is unavailable."""
    safe_prompt = requests.utils.quote(prompt)

    urls_to_try = [
        f"https://gen.pollinations.ai/image/{safe_prompt}?width=768&height=768",
        f"https://image.pollinations.ai/prompt/{safe_prompt}?width=768&height=768&nologo=true",
    ]

    last_error = None
    resp = None
    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            # confirm we actually got image bytes, not an error page
            content_type = resp.headers.get("Content-Type", "")
            if "image" in content_type or len(resp.content) > 1000:
                break
            resp = None
        except requests.RequestException as e:
            last_error = e
            resp = None
            continue

    if resp is None:
        raise RuntimeError(f"All Pollinations endpoints failed. Last error: {last_error}")

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(GENERATED_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(resp.content)

    entry = {
        "id": uuid.uuid4().hex,
        "type": "generated",
        "prompt": prompt,
        "filename": filename,
        "url": f"/static/generated/{filename}",
        "session_id": session_id,
        "owner_email": owner_email,
        "created_at": datetime.datetime.now().isoformat(),
    }
    add_to_image_index(entry)
    return entry


def save_uploaded_image(image_data_url, session_id, question="", owner_email=None):
    """Validate and save a small image data URL without trusting its filename/header."""
    if not isinstance(image_data_url, str):
        raise ValueError("Invalid image data.")
    try:
        header, b64data = image_data_url.split(",", 1)
    except ValueError:
        raise ValueError("Invalid image data URL.")

    header = header.lower()
    mime_ext = {
        "data:image/png": ("png", b"\x89PNG\r\n\x1a\n"),
        "data:image/jpeg": ("jpg", b"\xff\xd8\xff"),
        "data:image/jpg": ("jpg", b"\xff\xd8\xff"),
        "data:image/webp": ("webp", b"RIFF"),
    }
    match = next(((prefix, value) for prefix, value in mime_ext.items() if header.startswith(prefix)), None)
    if not match:
        raise ValueError("Unsupported image type. Use PNG, JPEG, or WebP.")

    ext, magic = match[1]
    try:
        raw = base64.b64decode(b64data, validate=True)
    except (ValueError, base64.binascii.Error):
        raise ValueError("Invalid base64 image data.")

    if not raw or len(raw) > 3 * 1024 * 1024:
        raise ValueError("Image is empty or too large. Max image size is 3MB.")
    if ext == "webp":
        if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
            raise ValueError("Invalid WebP image.")
    elif not raw.startswith(magic):
        raise ValueError("Image data does not match its declared type.")

    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOADED_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(raw)

    entry = {
        "id": uuid.uuid4().hex,
        "type": "uploaded",
        "prompt": question,
        "filename": filename,
        "url": f"/static/uploaded/{filename}",
        "session_id": session_id,
        "owner_email": owner_email,
        "created_at": datetime.datetime.now().isoformat(),
    }
    add_to_image_index(entry)
    return entry


def save_uploaded_file(file_data_url, original_name, session_id, owner_email=None):
    """Saves a base64 data URL file (non-image) to disk and extracts readable text if possible."""
    ext = (original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "").strip()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise ValueError(f"File type .{ext} isn't supported. Allowed: {', '.join(sorted(ALLOWED_FILE_EXTENSIONS))}")

    try:
        header, b64data = file_data_url.split(",", 1)
    except ValueError:
        b64data = file_data_url

    raw_bytes = base64.b64decode(b64data)
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")

    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_name)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    filepath = os.path.join(FILES_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(raw_bytes)

    text_content = None
    if ext in {"txt", "md", "csv", "json", "py", "js", "html", "css", "log"}:
        try:
            text_content = raw_bytes.decode("utf-8", errors="replace")
            if len(text_content) > 12000:
                text_content = text_content[:12000] + "\n\n[...truncated...]"
        except Exception:
            text_content = None

    register_private_file(filename, owner_email, session_id)
    return {
        "filename": filename,
        "original_name": original_name,
        "url": f"/static/files/{filename}",
        "owner_email": owner_email,
        "session_id": session_id,
        "text_content": text_content,
    }


# ---------- Auto file generation from AI replies ----------
# Maps a fenced code-block language hint to a sensible file extension.
LANGUAGE_TO_EXTENSION = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "html": "html", "css": "css",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "sql": "sql", "java": "java", "c": "c", "cpp": "cpp",
    "go": "go", "rust": "rs", "php": "php", "ruby": "rb",
    "markdown": "md", "md": "md", "csv": "csv", "xml": "xml",
}

# A reply is worth offering as a file when it clears any of these bars —
# tuned to catch "this is really a document/script" without flagging short answers.
MIN_CODEBLOCK_CHARS_FOR_FILE = 400   # a single code block this long is probably meant to be saved
MIN_TOTAL_CHARS_FOR_FILE = 1500      # a long reply overall (e.g. a big list, article, report)
MIN_LINES_FOR_FILE = 25              # or just a lot of lines, even if individually short


def detect_file_worthy_content(reply_text):
    """Looks at an AI reply and decides whether it's substantial enough to also
    offer as a downloadable file. Returns a dict with filename/content/mime, or None."""
    code_blocks = re.findall(r'```(\w*)\n(.*?)```', reply_text, re.DOTALL)

    # Case 1: one dominant code block that's long enough on its own
    if code_blocks:
        lang, code = max(code_blocks, key=lambda b: len(b[1]))
        code = code.strip()
        if len(code) >= MIN_CODEBLOCK_CHARS_FOR_FILE:
            ext = LANGUAGE_TO_EXTENSION.get(lang.lower().strip(), "txt")
            return {"content": code, "extension": ext, "reason": "code"}

    # Case 2: the whole reply is just long (article, report, big list) —
    # save the plain reply text as a markdown file.
    plain_length = len(reply_text)
    line_count = reply_text.count("\n") + 1
    if plain_length >= MIN_TOTAL_CHARS_FOR_FILE or line_count >= MIN_LINES_FOR_FILE:
        return {"content": reply_text, "extension": "md", "reason": "long_reply"}

    return None


def save_generated_reply_file(content, extension, session_id, owner_email=None):
    filename = f"{uuid.uuid4().hex}.{extension}"
    filepath = os.path.join(FILES_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    register_private_file(filename, owner_email, session_id)
    return f"/static/files/{filename}"


def estimate_context_chars(messages):
    return sum(len(str(m.get("content", ""))) for m in messages)

def build_bounded_messages(system_message, history, user_message, max_chars=MAX_CONTEXT_CHARS):
    """Keep the newest conversation context while enforcing a hard character budget."""
    system = system_message if isinstance(system_message, dict) else {"role": "system", "content": str(system_message)}
    selected = [{"role": "user", "content": user_message}]
    budget = max(4000, int(max_chars))
    used = len(str(system.get("content", ""))) + len(user_message)
    for item in reversed(history):
        content = str(item.get("content", ""))
        if not content:
            continue
        cost = len(content) + 32
        if used + cost > budget:
            break
        selected.insert(0, {"role": item.get("role", "user"), "content": content})
        used += cost
    return [system] + selected

def choose_ai_route(user_message, requested_model=None, settings=None):
    """Select a route without requiring the frontend to know provider details."""
    requested_model = requested_model or ((settings or {}).get("model") if settings else None) or CURRENT_MODEL
    text = (user_message or "").lower()
    complex_markers = ("debug", "analyze", "analysis", "prove", "algorithm", "architecture", "refactor", "why", "compare", "reason")
    route = "reasoning" if len(text) > 900 or any(x in text for x in complex_markers) else "fast"
    fast_model = os.environ.get("GUM_FAST_MODEL", requested_model)
    reasoning_model = os.environ.get("GUM_REASONING_MODEL", requested_model)
    model = reasoning_model if route == "reasoning" else fast_model
    provider = "openai" if model.startswith("gpt-") and AI_PROVIDERS["openai"]["available"] else "groq"
    if provider == "groq" and not AI_PROVIDERS["groq"]["available"]:
        provider = "openai"
    return provider, model, route

# ---------- Subagent orchestration ----------
# Gum can explicitly fan a task out to several independent AI workers, then
# have a lead agent synthesize their results. This is intentionally bounded so
# one chat cannot create an unbounded swarm or consume unlimited API calls.
SUBAGENT_MAX = max(1, min(int(os.environ.get("GUM_MAX_SUBAGENTS", "4")), 8))
SUBAGENT_TIMEOUT = max(10, min(int(os.environ.get("GUM_SUBAGENT_TIMEOUT", "90")), 180))
SUBAGENT_TRIGGER_RE = re.compile(r"(?:use|spawn|add|create|run)\\s+(?:up\\s+to\\s+)?(?:\\d+\\s+)?subagents?\\b", re.I)

SUBAGENT_ROLES = [
    ("researcher", "Investigate the task independently. Gather the key facts, assumptions, options, and useful evidence. Do not blindly trust the other workers."),
    ("planner", "Break the task into concrete steps and propose a practical solution. Call out dependencies, risks, and edge cases."),
    ("critic", "Act as a skeptical reviewer. Look for mistakes, missing requirements, contradictions, security issues, and weak assumptions."),
    ("builder", "Work toward an implementation-ready answer. Prefer concrete examples, code structure, tests, and actionable next steps where relevant."),
    ("designer", "Act as Gum's product and UI/UX designer. Turn requirements into clear user experiences, screen layouts, interaction flows, accessibility considerations, responsive/mobile-first design guidance, and visual/component recommendations. When useful, provide implementation-ready HTML/CSS/component structure without sacrificing usability."),
]

# Autonomous subagent mode. When enabled, Gum may decide that a request is
# complex enough to benefit from specialist workers even when the user did not
# explicitly ask for them. The decision is bounded and never creates an
# unbounded recursive swarm.
AUTO_SUBAGENTS = os.environ.get("GUM_AUTO_SUBAGENTS", "true").lower() in ("1", "true", "yes", "on")
AUTO_SUBAGENT_MIN_CHARS = max(120, min(int(os.environ.get("GUM_AUTO_SUBAGENT_MIN_CHARS", "500")), 4000))
AUTO_SUBAGENT_MAX = max(1, min(int(os.environ.get("GUM_AUTO_SUBAGENT_MAX", str(SUBAGENT_MAX))), SUBAGENT_MAX))

def _looks_complex_enough(text):
    text = (text or "").strip()
    if len(text) < AUTO_SUBAGENT_MIN_CHARS:
        return False
    complexity_terms = (
        "build", "design", "architect", "compare", "analyze", "research",
        "debug", "refactor", "plan", "implement", "review", "strategy",
        "multiple", "step by step", "tradeoff", "requirements", "system",
        "website", "app", "code", "project", "security", "database"
    )
    return sum(1 for term in complexity_terms if term in text.lower()) >= 2 or len(text) >= 1200

def gum_wants_subagents(text, history, model, email, session_id, temperature):
    """Let Gum make a bounded yes/no decision for complex requests."""
    if not AUTO_SUBAGENTS or not _looks_complex_enough(text):
        return False, min(3, AUTO_SUBAGENT_MAX), text

    decision_prompt = (
        "You are Gum's orchestration controller. Decide whether this user request "
        "would materially benefit from independent specialist subagents. Spawn them "
        "only when parallel perspectives or specialized work would improve the result. "
        "Do not spawn for simple questions, casual chat, short factual answers, or tasks "
        "where one agent is clearly sufficient. Return ONLY compact JSON: "
        '{"spawn":true|false,"count":1-4,"task":"..."}.\n\n'
        f"USER REQUEST:\n{text}\n\n"
        f"RECENT CONTEXT:\n{json.dumps(history[-4:], ensure_ascii=False)[:5000]}"
    )
    try:
        response, _, _, _ = ai_complete(
            [{"role": "system", "content": "You are a conservative task-routing controller."},
             {"role": "user", "content": decision_prompt}],
            email, session_id, text, model, 0.0
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.S)
        if not match:
            return False, min(3, AUTO_SUBAGENT_MAX), text
        decision = json.loads(match.group(0))
        spawn = bool(decision.get("spawn"))
        count = max(1, min(int(decision.get("count", 3)), AUTO_SUBAGENT_MAX))
        task = str(decision.get("task") or text).strip()[:12000]
        return spawn, count, task
    except Exception:
        # A routing failure must never prevent the normal single-agent response.
        return False, min(3, AUTO_SUBAGENT_MAX), text

def subagent_requested(text):
    return bool(SUBAGENT_TRIGGER_RE.search(text or "")) or str(text or "").strip().lower().startswith("/subagents")

def _subagent_count_and_task(text):
    raw = (text or "").strip()
    if raw.lower().startswith("/subagents"):
        raw = raw[len("/subagents"):].strip()
    m = re.match(r"^(\\d+)\\s*[:,-]?\\s*(.*)$", raw, re.S)
    if m:
        count = max(1, min(int(m.group(1)), SUBAGENT_MAX))
        task = m.group(2).strip()
    else:
        count = min(3, SUBAGENT_MAX)
        task = raw
    return count, task or "Solve the user's request independently."

def run_subagents(task, email, session_id, requested_model=None, temperature=0.7, count=None):
    """Run bounded independent workers in parallel and return their findings."""
    count = max(1, min(int(count or 3), SUBAGENT_MAX))
    jobs = [(SUBAGENT_ROLES[i % len(SUBAGENT_ROLES)]) for i in range(count)]

    def worker(item):
        role, instructions = item
        prompt = (
            f"You are Gum subagent #{jobs.index(item)+1}, role: {role}.\\n"
            f"Your job: {instructions}\\n\\n"
            f"Main task:\n{task}\\n\\n"
            "Return concise, useful findings for a lead agent. Do not claim you performed actions you cannot actually perform."
        )
        sys_msg = {"role": "system", "content": "You are an independent specialist subagent inside Gum."}
        msgs = [sys_msg, {"role": "user", "content": prompt}]
        response, provider, model, route = ai_complete(
            msgs, email, session_id, task, requested_model, temperature
        )
        return {"role": role, "reply": response.choices[0].message.content, "provider": provider, "model": model, "route": route}

    results = []
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"role": "worker-error", "reply": f"Subagent failed: {type(exc).__name__}: {exc}"})
    return results

def synthesize_subagents(task, findings, email, session_id, requested_model=None, temperature=0.7):
    """Have a lead AI turn independent worker findings into one user-facing answer."""
    joined = "\n\n".join(
        f"[{item.get('role','subagent').upper()}]\n{item.get('reply','')}"
        for item in findings
    )
    prompt = (
        "You are Gum's lead agent. Synthesize the independent subagent reports below into one "
        "accurate answer to the main task. Resolve disagreements instead of averaging them. "
        "Do not mention internal orchestration unless useful to the user. If evidence is missing, "
        "say so rather than inventing it.\n\n"
        f"MAIN TASK:\n{task}\n\nSUBAGENT REPORTS:\n{joined}"
    )
    sys_msg = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT + "\nYou are the lead agent coordinating specialist subagents."}
    response, provider, model, route = ai_complete(
        [sys_msg, {"role": "user", "content": prompt}],
        email, session_id, task, requested_model, temperature
    )
    return response.choices[0].message.content, {"provider": provider, "model": model, "route": route}

def ai_complete(messages, email, session_id, user_message, requested_model=None, temperature=1.0, settings=None):
    """Provider-aware completion with automatic routing, one safe fallback, and telemetry."""
    provider, model, route = choose_ai_route(user_message, requested_model, settings)
    candidates = [(provider, model, route)]
    fallback_provider = "openai" if provider == "groq" else "groq"
    fallback_model = os.environ.get("GUM_OPENAI_FALLBACK_MODEL", "gpt-4o-mini") if fallback_provider == "openai" else os.environ.get("GUM_GROQ_FALLBACK_MODEL", CURRENT_MODEL)
    if AI_PROVIDERS.get(fallback_provider, {}).get("available") and (fallback_provider, fallback_model) != (provider, model):
        candidates.append((fallback_provider, fallback_model, "fallback"))

    last_error = None
    for idx, (candidate_provider, candidate_model, candidate_route) in enumerate(candidates):
        started = time.perf_counter()
        try:
            response = call_ai_provider(candidate_provider, candidate_model, messages, temperature=temperature)
            latency = int((time.perf_counter() - started) * 1000)
            usage = getattr(response, "usage", None)
            usage_dict = {}
            if usage is not None:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = getattr(usage, key, None)
                    if value is not None:
                        usage_dict[key] = value
            record_ai_usage(email, session_id, candidate_provider, candidate_model, candidate_route, latency, usage_dict, True)
            return response, candidate_provider, candidate_model, candidate_route
        except Exception as exc:
            last_error = exc
            latency = int((time.perf_counter() - started) * 1000)
            record_ai_usage(email, session_id, candidate_provider, candidate_model, candidate_route, latency, {}, False, type(exc).__name__)
            if idx + 1 < len(candidates):
                continue
    raise last_error or RuntimeError("No AI provider is available.")

def ask_once(prompt_text, model=None, system_prompt=None):
    sys_msg = {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
    msgs = [sys_msg, {"role": "user", "content": prompt_text}]
    try:
        response = call_ai_provider("groq", model or CURRENT_MODEL, msgs)
        return response.choices[0].message.content
    except Exception as e:
        return f"Something went wrong: {e}"


def get_history(session_id, owner_email=None, public_session_id=None):
    if session_id not in conversations:
        loaded = load_memory_chat(owner_email, public_session_id) if owner_email and public_session_id else None
        conversations[session_id] = (loaded or {}).get("messages", []) if loaded else []
    return conversations[session_id]


def is_premium(session_id):
    return premium_sessions.get(session_id, False)


def try_unlock_premium(session_id, code):
    if code and code.strip() == PREMIUM_CODE:
        premium_sessions[session_id] = True
        return True
    return False


# ---------- Safe calculator ----------
_CALC_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALC_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CALC_FUNCS = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
_CALC_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}

def safe_calculate(expression, max_nodes=80):
    """Evaluate a small mathematical expression without executing Python code."""
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > max_nodes:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(float(node.value)): raise ValueError("Non-finite number")
            return node.value
        if isinstance(node, ast.Name) and node.id in _CALC_NAMES:
            return _CALC_NAMES[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARY:
            return _CALC_UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _CALC_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCS and not node.keywords:
            if len(node.args) > 3: raise ValueError("Too many arguments")
            return _CALC_FUNCS[node.func.id](*[visit(a) for a in node.args])
        raise ValueError("Unsupported expression")

    result = visit(tree)
    if isinstance(result, (int, float)) and not math.isfinite(float(result)):
        raise ValueError("Non-finite result")
    return result


# ---------- Command handling ----------
def handle_command(cmd_raw, session_id, model, system_prompt=None, owner_email=None):
    """Returns a reply string for any slash command. Returns None if not a recognized command."""
    cmd = cmd_raw.strip()
    lower = cmd.lower()

    if lower in ("/help", "/commands"):
        return (
            "Commands:\n"
            "/joke /fact /quote — AI generated fun\n"
            "/flip — coin flip\n"
            "/roll N — random number 1-N\n"
            "/choose a,b,c — pick one randomly\n"
            "/password N — random password of length N\n"
            "/calc expr — calculator (e.g. /calc 12*7)\n"
            "/hash text — MD5 + SHA256 of text\n"
            "/upper text, /lower text, /reverse text\n"
            "/wrap text — wrap text to 40 chars\n"
            "/translate lang text — translate\n"
            "/summarize text — summarize\n"
            "/explain code — explain code\n"
            "/fix code — find/fix bugs\n"
            "/rhyme word — rhyming words\n"
            "/define word — definition\n"
            "/note text — save a note\n"
            "/notes — list saved notes\n"
            "/clearnotes — delete all notes\n"
            "/time — current date/time\n"
            "/calendar — this month's calendar\n"
            "/sysinfo — server OS/Python info\n"
            "/wordcount text — count words\n"
            "/charcount text — count characters\n"
            "/version — app version\n"
            "/imagine prompt — generate an image from text\n"
            "/clear — clear this conversation's memory"
        )

    if lower == "/joke":
        return ask_once("Tell me a short, clean joke", model)
    if lower == "/fact":
        return ask_once("Tell me one interesting random fact", model, system_prompt)
    if lower == "/quote":
        return ask_once("Give me an inspiring quote", model, system_prompt)

    if lower == "/flip":
        return f"**{random.choice(['Heads', 'Tails'])}**"

    if lower.startswith("/roll"):
        parts = cmd.split(" ", 1)
        try:
            n = int(parts[1]) if len(parts) > 1 else 100
            return f"You rolled: **{random.randint(1, n)}**"
        except ValueError:
            return "Usage: /roll 100"

    if lower.startswith("/choose "):
        options = cmd.split(" ", 1)[1].split(",")
        return f"Chosen: **{random.choice(options).strip()}**"

    if lower.startswith("/password"):
        parts = cmd.split(" ", 1)
        try:
            length = int(parts[1]) if len(parts) > 1 else 16
            chars = string.ascii_letters + string.digits + "!@#$%^&*"
            return f"Password: `{''.join(random.choice(chars) for _ in range(length))}`"
        except ValueError:
            return "Usage: /password 16"

    if lower.startswith("/calc "):
        expr = cmd.split(" ", 1)[1]
        try:
            result = safe_calculate(expr)
            return f"Result: **{result}**"
        except Exception:
            return "Invalid expression."

    if lower.startswith("/hash "):
        text = cmd.split(" ", 1)[1]
        return (f"MD5: `{hashlib.md5(text.encode()).hexdigest()}`\n"
                f"SHA256: `{hashlib.sha256(text.encode()).hexdigest()}`")

    if lower.startswith("/upper "):
        return cmd.split(" ", 1)[1].upper()
    if lower.startswith("/lower "):
        return cmd.split(" ", 1)[1].lower()
    if lower.startswith("/reverse "):
        return cmd.split(" ", 1)[1][::-1]
    if lower.startswith("/wrap "):
        return textwrap.fill(cmd.split(" ", 1)[1], width=40)

    if lower.startswith("/translate "):
        parts = cmd.split(" ", 2)
        if len(parts) < 3:
            return "Usage: /translate spanish hello there"
        return ask_once(f"Translate this to {parts[1]}: {parts[2]}", model, system_prompt)

    if lower.startswith("/summarize "):
        text = cmd.split(" ", 1)[1]
        return ask_once(f"Summarize this concisely: {text}", model, system_prompt)

    if lower.startswith("/explain "):
        code = cmd.split(" ", 1)[1]
        return ask_once(f"Explain this code simply:\n{code}", model, system_prompt)

    if lower.startswith("/fix "):
        code = cmd.split(" ", 1)[1]
        return ask_once(f"Find and fix bugs in this code:\n{code}", model, system_prompt)

    if lower.startswith("/rhyme "):
        word = cmd.split(" ", 1)[1]
        return ask_once(f"Give me 5 words that rhyme with {word}", model, system_prompt)

    if lower.startswith("/define "):
        word = cmd.split(" ", 1)[1]
        return ask_once(f"Define the word: {word}", model, system_prompt)

    if lower.startswith("/note "):
        text = cmd.split(" ", 1)[1]
        notes_store.append({"text": text, "time": str(datetime.datetime.now())})
        save_notes()
        return "Note saved."

    if lower == "/notes":
        if not notes_store:
            return "No notes saved."
        return "\n".join(f"{i+1}. {n['text']} ({n['time']})" for i, n in enumerate(notes_store))

    if lower == "/clearnotes":
        notes_store.clear()
        save_notes()
        return "All notes cleared."

    if lower == "/time":
        return str(datetime.datetime.now())

    if lower == "/calendar":
        now = datetime.datetime.now()
        return f"```\n{calendar.month(now.year, now.month)}\n```"

    if lower == "/sysinfo":
        return f"OS: {platform.system()} {platform.release()}\nPython: {platform.python_version()}"

    if lower.startswith("/wordcount "):
        text = cmd.split(" ", 1)[1]
        return f"Word count: {len(text.split())}"

    if lower.startswith("/charcount "):
        text = cmd.split(" ", 1)[1]
        return f"Characters: {len(text)}"

    if lower == "/version":
        return f"{DISPLAY_NAME} web — v{APP_VERSION}, powered by Groq + OpenAI when configured"

    if lower.startswith("/imagine "):
        prompt = cmd.split(" ", 1)[1]
        try:
            entry = generate_image(prompt, session_id, owner_email)
            return f"__IMAGE__{entry['url']}__CAPTION__{prompt}"
        except Exception as e:
            return f"Image generation failed: {e}"

    if lower == "/clear":
        conversations[session_id] = []
        return "__CLEAR__"  # special signal handled by caller

    return None  # not a recognized command


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{{ display_name }} — a fast, friendly AI assistant with chat, image generation, plugins, and more.">
<meta name="theme-color" content="#211F1D">
<meta property="og:title" content="{{ display_name }}">
<meta property="og:description" content="Chat, generate images, build plugins, and more.">
<title>{{ display_name }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>

  /* ============================================================
     GUM — design tokens
     Warm chalk-and-charcoal identity. One bold move: bubble shape
     language + the send button's press state. Everything else is
     quiet, consistent, and gets out of the way.
     ============================================================ */

  :root {
    --charcoal: #211F1D;
    --charcoal-deep: #171615;
    --charcoal-raised: #2A2725;
    --chalk: #F2ECDC;
    --chalk-dim: #C9C2AE;
    --amber: #E39A5C;
    --amber-dim: #B87A44;
    --amber-glow: rgba(227, 154, 92, 0.18);
    --sage: #5F9382;
    --sage-dim: #4A7666;
    --sage-glow: rgba(95, 147, 130, 0.16);
    --line: #3A3634;
    --line-soft: rgba(58, 54, 52, 0.55);
    --danger: #D97757;
    --shadow-soft: 0 2px 10px rgba(0, 0, 0, 0.18);
    --shadow-lifted: 0 12px 40px rgba(0, 0, 0, 0.35);
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
    --base-font-size: 15px;
    --radius-md: 14px;
    --radius-sm: 10px;
  }

  [data-theme="forest"] {
    --charcoal: #17211C;
    --charcoal-deep: #101915;
    --charcoal-raised: #1E2A23;
    --chalk: #EAF0EB;
    --chalk-dim: #B9C8BE;
    --amber: #A8CBA9;
    --amber-dim: #86AD88;
    --amber-glow: rgba(168, 203, 169, 0.16);
    --sage: #D8A85C;
    --sage-dim: #B98C48;
    --sage-glow: rgba(216, 168, 92, 0.16);
    --line: #2E3A32;
    --line-soft: rgba(46, 58, 50, 0.55);
  }

  [data-theme="mono"] {
    --charcoal: #202020;
    --charcoal-deep: #151515;
    --charcoal-raised: #292929;
    --chalk: #EDEDED;
    --chalk-dim: #ABABAB;
    --amber: #EDEDED;
    --amber-dim: #B3B3B3;
    --amber-glow: rgba(237, 237, 237, 0.1);
    --sage: #9A9A9A;
    --sage-dim: #7A7A7A;
    --sage-glow: rgba(154, 154, 154, 0.12);
    --line: #333333;
    --line-soft: rgba(51, 51, 51, 0.55);
  }

  body[data-font="small"] { --base-font-size: 13.5px; }
  body[data-font="medium"] { --base-font-size: 15px; }
  body[data-font="large"] { --base-font-size: 17px; }

  * { box-sizing: border-box; }

  ::selection { background: var(--amber-glow); color: var(--chalk); }

  *:focus-visible {
    outline: 2px solid var(--amber);
    outline-offset: 2px;
    border-radius: 4px;
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.001ms !important;
      transition-duration: 0.001ms !important;
    }
  }

  html { color-scheme: dark; }

  body {
    margin: 0;
    min-height: 100vh;
    background:
      radial-gradient(ellipse 900px 600px at 12% 0%, rgba(227, 154, 92, 0.05), transparent 55%),
      radial-gradient(ellipse 900px 700px at 90% 100%, rgba(95, 147, 130, 0.06), transparent 55%),
      var(--charcoal);
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--chalk);
    -webkit-font-smoothing: antialiased;
  }

  /* ------------------------------------------------------------
     Layout
     ------------------------------------------------------------ */
  .layout {
    display: flex;
    min-height: 100vh;
  }

  #sidebar {
    width: 272px;
    flex-shrink: 0;
    background: var(--charcoal-deep);
    border-right: 1px solid var(--line-soft);
    display: flex;
    flex-direction: column;
    padding: 20px 16px 16px;
    gap: 4px;
    height: 100vh;
    position: sticky;
    top: 0;
    overflow-y: auto;
    scrollbar-width: thin;
  }

  .sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
    padding: 0 2px;
  }

  .sidebar-header #closeSidebar { display: none; }

  .brand {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 26px;
    letter-spacing: -0.03em;
    color: var(--chalk);
    line-height: 1;
  }

  .brand span { color: var(--amber); }
  .brand.small { font-size: 18px; }

  .side-btn.primary {
    background: var(--amber);
    color: var(--charcoal-deep);
    border: none;
    border-radius: var(--radius-sm);
    padding: 11px 0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    margin-bottom: 22px;
    transition: transform 0.15s var(--ease), background 0.15s ease;
  }

  .side-btn.primary:hover { background: #eaa76a; }
  .side-btn.primary:active { transform: scale(0.97); }

  .side-section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11.5px;
    font-weight: 500;
    color: var(--chalk-dim);
    margin: 18px 2px 9px;
  }

  .quick-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
  }

  .quick-btn {
    background: var(--charcoal-raised);
    border: 1px solid transparent;
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 10px 10px;
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s ease, transform 0.1s ease;
  }

  .quick-btn:hover { border-color: var(--line); }
  .quick-btn:active { transform: scale(0.97); }

  .saved-chats-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    overflow-y: auto;
  }

  .saved-chats-empty {
    font-size: 12.5px;
    color: var(--chalk-dim);
    padding: 10px 4px;
  }

  .saved-chat-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 10px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background 0.12s ease;
  }

  .saved-chat-item:hover { background: var(--charcoal-raised); }

  .saved-chat-title {
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  }

  .saved-chat-delete {
    background: none;
    border: none;
    color: var(--chalk-dim);
    cursor: pointer;
    font-size: 12px;
    padding: 2px 6px;
    opacity: 0;
    transition: opacity 0.12s ease;
  }

  .saved-chat-item:hover .saved-chat-delete { opacity: 1; }
  .saved-chat-delete:hover { color: var(--danger); }

  #sidebarToggle {
    display: none;
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 40;
    background: var(--charcoal-raised);
    box-shadow: var(--shadow-soft);
  }

  .app {
    width: 100%;
    max-width: 680px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 26px 20px 20px;
  }

  /* ------------------------------------------------------------
     Header
     ------------------------------------------------------------ */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 2px 18px;
    border-bottom: 1px solid var(--line-soft);
    margin-bottom: 18px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  select {
    background: var(--charcoal-raised);
    color: var(--chalk);
    border: 1px solid var(--line);
    border-radius: 9px;
    padding: 7px 10px;
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    max-width: 150px;
    cursor: pointer;
  }

  .icon-btn {
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: 9px;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 15px;
    transition: border-color 0.15s ease, transform 0.1s ease;
  }

  .icon-btn:hover { border-color: var(--amber-dim); }
  .icon-btn:active { transform: scale(0.93); }
  .account-btn.logged-in { color: var(--amber); border-color: var(--amber-dim); }

  /* ------------------------------------------------------------
     Chat feed
     ------------------------------------------------------------ */
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 4px 2px 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    scroll-behavior: smooth;
  }

  .row {
    display: flex;
    flex-direction: column;
    max-width: 80%;
    animation: rise 0.22s var(--ease);
  }

  @keyframes rise {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .row.user { align-self: flex-end; align-items: flex-end; }
  .row.gum { align-self: flex-start; align-items: flex-start; }
  .row.system { align-self: center; align-items: center; max-width: 95%; }

  .label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: var(--chalk-dim);
    margin-bottom: 6px;
    padding: 0 2px;
  }

  .row.system .label { opacity: 0.6; }

  .bubble {
    padding: 12px 16px;
    line-height: 1.55;
    font-size: var(--base-font-size);
    white-space: pre-wrap;
    box-shadow: var(--shadow-soft);
  }

  /* Bold move: asymmetric radii — a soft corner and a sharp corner
     pointing toward the speaker, instead of uniform rounded-rect bubbles */
  .row.user .bubble {
    background: var(--amber);
    color: var(--charcoal-deep);
    border-radius: 16px 16px 4px 16px;
  }

  .row.gum .bubble {
    background: var(--charcoal-raised);
    border: 1px solid var(--line-soft);
    color: var(--chalk);
    border-radius: 16px 16px 16px 4px;
    box-shadow: none;
  }

  .row.system .bubble {
    background: transparent;
    border: 1px dashed var(--line);
    color: var(--chalk-dim);
    font-size: 12.5px;
    text-align: left;
    border-radius: var(--radius-sm);
    max-width: 100%;
    overflow-x: auto;
  }

  .bubble code {
    background: rgba(0,0,0,0.28);
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 0.87em;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    word-break: break-all;
    overflow-wrap: anywhere;
    display: inline-block;
    max-width: 100%;
  }

  .row.user .bubble code { background: rgba(0,0,0,0.15); }

  .bubble pre {
    background: var(--charcoal-deep);
    border: 1px solid var(--line-soft);
    padding: 13px 14px;
    border-radius: var(--radius-sm);
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    margin: 10px 0 0;
    max-width: 100%;
  }

  .bubble pre code {
    background: none;
    padding: 0;
    display: inline;
    white-space: pre;
    word-break: normal;
    overflow-wrap: normal;
  }

  /* Breathing indicator instead of bouncing dots */
  .thinking {
    display: flex;
    gap: 5px;
    padding: 15px 16px;
    align-items: center;
  }

  .thinking span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sage);
    animation: breathe 1.6s ease-in-out infinite;
  }

  .thinking span:nth-child(2) { animation-delay: 0.2s; }
  .thinking span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes breathe {
    0%, 100% { opacity: 0.25; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1.05); }
  }

  .hint {
    font-size: 12px;
    color: var(--chalk-dim);
    padding: 0 4px 12px;
  }

  .hint code {
    background: var(--charcoal-raised);
    border: 1px solid var(--line-soft);
    padding: 1px 6px;
    border-radius: 5px;
  }

  .chat-image {
    max-width: 100%;
    border-radius: var(--radius-sm);
    display: block;
    margin-top: 4px;
    box-shadow: var(--shadow-soft);
  }

  .file-offer {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    padding: 9px 14px;
    background: var(--charcoal-raised);
    border: 1px solid var(--amber-dim);
    border-radius: var(--radius-sm);
    color: var(--amber);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 12.5px;
    text-decoration: none;
    transition: background 0.15s ease;
  }

  .file-offer:hover { background: var(--amber-glow); }

  /* ------------------------------------------------------------
     Composer
     ------------------------------------------------------------ */
  form#composer {
    display: flex;
    gap: 10px;
    padding-top: 16px;
    border-top: 1px solid var(--line-soft);
  }

  .plus-btn {
    align-self: stretch;
    width: 46px;
    font-size: 22px;
    font-weight: 300;
    line-height: 1;
  }

  /* ---------- Model picker (composer) ---------- */
  #modelPicker {
    position: relative;
    align-self: stretch;
    flex-shrink: 0;
  }

  #modelPickerBtn {
    height: 100%;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 0 12px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12.5px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    white-space: nowrap;
    max-width: 108px;
    transition: border-color 0.15s ease;
  }

  #modelPickerBtn:hover { border-color: var(--amber-dim); }
  #modelPickerBtn.open { border-color: var(--amber); }

  #modelPickerCurrent {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .model-picker-caret {
    font-size: 10px;
    opacity: 0.6;
    flex-shrink: 0;
    transition: transform 0.15s ease;
  }

  #modelPickerBtn.open .model-picker-caret { transform: rotate(180deg); }

  #modelPickerMenu {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    min-width: 240px;
    max-width: 80vw;
    max-height: 320px;
    overflow-y: auto;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lifted);
    padding: 6px;
    display: none;
    flex-direction: column;
    gap: 2px;
    z-index: 30;
    animation: riseIn 0.15s var(--ease);
  }

  #modelPickerMenu.open { display: flex; }

  .model-option {
    background: none;
    border: none;
    color: var(--chalk);
    text-align: left;
    padding: 9px 11px;
    border-radius: 9px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 2px;
    transition: background 0.12s ease;
  }

  .model-option:hover { background: var(--charcoal-raised); }
  .model-option.active { background: var(--amber-glow); }

  .model-option-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13px;
  }

  .model-option.active .model-option-name { color: var(--amber); }

  .model-option-desc {
    font-size: 11.5px;
    color: var(--chalk-dim);
  }

  @media (max-width: 480px) {
    #modelPickerBtn { max-width: 86px; padding: 0 9px; font-size: 12px; }
    #modelPickerMenu { min-width: 200px; }
  }

  #userInput {
    flex: 1;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 13px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    resize: none;
    outline: none;
    transition: border-color 0.15s ease;
  }

  #userInput:focus { border-color: var(--amber-dim); }
  #userInput::placeholder { color: var(--chalk-dim); }

  #sendBtn {
    background: var(--amber);
    border: none;
    color: var(--charcoal-deep);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
    border-radius: var(--radius-sm);
    padding: 0 24px;
    cursor: pointer;
    transition: transform 0.12s var(--ease), background 0.15s ease;
  }

  /* Bold move: the send button has a real tactile press */
  #sendBtn:hover { background: #eaa76a; }
  #sendBtn:active { transform: scale(0.94); }
  #sendBtn:disabled { opacity: 0.45; cursor: default; transform: none; }

  #chat::-webkit-scrollbar,
  #sidebar::-webkit-scrollbar,
  .plugins-grid::-webkit-scrollbar,
  .gallery-grid::-webkit-scrollbar {
    width: 6px;
  }

  #chat::-webkit-scrollbar-thumb,
  #sidebar::-webkit-scrollbar-thumb {
    background: var(--line);
    border-radius: 4px;
  }

  /* ------------------------------------------------------------
     Overlays — shared foundation
     ------------------------------------------------------------ */
  #settingsOverlay, #galleryOverlay, #pluginsOverlay, #lockScreen,
  #attachMenuOverlay, #promptModalOverlay, #authOverlay {
    position: fixed;
    inset: 0;
    background: rgba(10, 9, 8, 0.6);
    backdrop-filter: blur(3px);
    display: none;
    z-index: 50;
  }

  #settingsOverlay.open, #galleryOverlay.open, #pluginsOverlay.open,
  #lockScreen.open, #attachMenuOverlay.open, #promptModalOverlay.open,
  #authOverlay.open {
    display: flex;
  }

  #settingsOverlay {
    justify-content: flex-end;
    align-items: stretch;
  }

  #settingsPanel {
    width: 360px;
    max-width: 90vw;
    height: 100vh;
    height: 100dvh; /* real fix: dvh tracks the actual visible viewport on mobile, unlike 100% against a flex parent */
    background: var(--charcoal);
    border-left: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: slideInRight 0.22s var(--ease);
  }

  @keyframes slideInRight {
    from { transform: translateX(24px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }

  .settings-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 22px;
    border-bottom: 1px solid var(--line-soft);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 16px;
    flex-shrink: 0;
  }

  .settings-body {
    padding: 20px 22px;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    flex: 1 1 auto;
    min-height: 0; /* the actual fix: without this, a flex child can't shrink below its content size, so it never scrolls */
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .settings-body label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: var(--chalk-dim);
    margin-top: 18px;
    display: block;
  }

  .settings-body label:first-child { margin-top: 0; }
  .settings-body label .sub { font-weight: 400; color: var(--chalk-dim); opacity: 0.85; }

  .theme-row { display: flex; gap: 10px; margin-top: 8px; }

  .theme-swatch {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }

  .theme-swatch:hover { transform: scale(1.08); }
  .theme-swatch.dusk { background: linear-gradient(135deg, #211F1D, #E39A5C); }
  .theme-swatch.forest { background: linear-gradient(135deg, #17211C, #D8A85C); }
  .theme-swatch.mono { background: linear-gradient(135deg, #202020, #EDEDED); }
  .theme-swatch.active { border-color: var(--chalk); }

  .segmented {
    display: flex;
    margin-top: 8px;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--charcoal-deep);
  }

  .segmented button {
    flex: 1;
    background: none;
    border: none;
    color: var(--chalk-dim);
    padding: 9px 0;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease;
  }

  .segmented button.active {
    background: var(--charcoal-raised);
    color: var(--chalk);
  }

  input[type="range"] {
    margin-top: 10px;
    width: 100%;
    accent-color: var(--amber);
  }

  .range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--chalk-dim);
    margin-top: 3px;
  }

  #systemPromptBox {
    margin-top: 8px;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 11px 13px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    line-height: 1.5;
    resize: vertical;
    outline: none;
    width: 100%;
    height: 140px;
    min-height: 140px;
    max-height: 320px;
    flex-shrink: 0;
    box-sizing: border-box;
    overflow-y: scroll;
    -webkit-overflow-scrolling: touch;
    display: block;
  }

  .settings-actions {

    display: flex;
    gap: 10px;
    margin-top: 24px;
  }

  .settings-actions button {
    flex: 1;
    padding: 11px 0;
    border-radius: var(--radius-sm);
    border: none;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    transition: transform 0.12s ease;
  }

  .settings-actions button:active { transform: scale(0.97); }

  #saveSettings { background: var(--amber); color: var(--charcoal-deep); }
  .settings-actions .ghost, .ghost {
    background: none;
    border: 1px solid var(--line);
    color: var(--chalk);
  }

  .ghost.small {
    font-size: 11.5px;
    padding: 7px 11px;
    border-radius: 8px;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    cursor: pointer;
    transition: border-color 0.15s ease;
  }

  .ghost.small:hover { border-color: var(--amber-dim); }

  /* ------------------------------------------------------------
     Backgrounds & color wheel
     ------------------------------------------------------------ */
  .bg-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .bg-swatch.none {
    width: 44px;
    height: 44px;
    border-radius: var(--radius-sm);
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk-dim);
    cursor: pointer;
    flex-shrink: 0;
    font-size: 14px;
    transition: border-color 0.15s ease;
  }

  .bg-swatch.none.active { border-color: var(--amber); color: var(--amber); }

  .bg-picker-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    flex: 1;
  }

  .bg-thumb {
    width: 44px;
    height: 44px;
    border-radius: var(--radius-sm);
    object-fit: cover;
    cursor: pointer;
    border: 2px solid transparent;
    opacity: 0.7;
    transition: opacity 0.15s ease, border-color 0.15s ease;
  }

  .bg-thumb:hover { opacity: 1; }
  .bg-thumb.active { border-color: var(--amber); opacity: 1; }

  .bg-picker-empty {
    font-size: 12px;
    color: var(--chalk-dim);
    align-self: center;
  }

  .wheel-row {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-top: 10px;
  }

  #colorWheel {
    position: relative;
    width: 116px;
    height: 116px;
    border-radius: 50%;
    background: conic-gradient(red, yellow, lime, cyan, blue, magenta, red);
    box-shadow: 0 0 0 1px var(--line), var(--shadow-soft);
    cursor: pointer;
    flex-shrink: 0;
    touch-action: none;
  }

  #colorWheel::after {
    content: "";
    position: absolute;
    inset: 38%;
    border-radius: 50%;
    background: var(--charcoal);
  }

  #colorWheelHandle {
    position: absolute;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid var(--charcoal-deep);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    box-shadow: 0 1px 5px rgba(0,0,0,0.5);
  }

  .wheel-info {
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: flex-start;
  }

  #colorSwatchPreview {
    width: 38px;
    height: 38px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--line);
    background: var(--chalk);
  }

  body.has-bg-image #chat {
    background-size: cover;
    background-position: center;
    border-radius: var(--radius-md);
    padding: 14px;
  }

  body.has-bg-image .row.gum .bubble,
  body.has-bg-image .row.user .bubble {
    backdrop-filter: blur(3px);
  }

  /* ------------------------------------------------------------
     Gallery
     ------------------------------------------------------------ */
  #galleryOverlay, #pluginsOverlay, #searchOverlay {
    align-items: center;
    justify-content: center;
    padding: 20px;
  }

  #galleryPanel {
    width: 100%;
    max-width: 720px;
    max-height: 82vh;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: var(--shadow-lifted);
    animation: riseIn 0.2s var(--ease);
  }

  /* ---------- Search ---------- */
  #searchOverlay {
    position: fixed;
    inset: 0;
    background: rgba(10, 9, 8, 0.6);
    backdrop-filter: blur(3px);
    display: none;
    z-index: 65;
  }

  #searchOverlay.open { display: flex; }

  #searchPanel {
    width: 100%;
    max-width: 460px;
    max-height: 70vh;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: var(--shadow-lifted);
    animation: riseIn 0.18s var(--ease);
  }

  #searchInput {
    width: 100%;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    font-size: 14px;
    outline: none;
    box-sizing: border-box;
  }

  #searchResults {
    overflow-y: auto;
    padding: 0 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .search-result-item {
    background: var(--charcoal-raised);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    cursor: pointer;
    transition: border-color 0.15s ease;
  }

  .search-result-item:hover { border-color: var(--amber-dim); }

  .search-result-label {
    display: block;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--chalk-dim);
    margin-bottom: 3px;
  }

  .search-result-snippet {
    font-size: 13px;
    color: var(--chalk);
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .search-empty {
    padding: 30px 20px;
    text-align: center;
    color: var(--chalk-dim);
    font-size: 13px;
  }

  .row.search-highlight .bubble {
    outline: 2px solid var(--amber);
    outline-offset: 2px;
  }

  /* ---------- Mic button ---------- */
  #micBtn.listening {
    border-color: var(--danger);
    animation: micPulse 1s ease-in-out infinite;
  }

  @keyframes micPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  @keyframes riseIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }

  .gallery-grid {
    padding: 18px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
  }

  .gallery-item {
    position: relative;
    border-radius: var(--radius-sm);
    overflow: hidden;
    aspect-ratio: 1;
    border: 1px solid var(--line-soft);
  }

  .gallery-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .gallery-item .gallery-caption {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.8));
    color: #fff;
    font-size: 10.5px;
    padding: 14px 7px 7px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .gallery-empty, .plugins-empty {
    padding: 48px 20px;
    text-align: center;
    color: var(--chalk-dim);
    font-size: 13px;
    grid-column: 1 / -1;
  }

  /* ------------------------------------------------------------
     Plugins
     ------------------------------------------------------------ */
  #pluginsPanel {
    width: 100%;
    max-width: 840px;
    height: 85vh;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: var(--shadow-lifted);
    animation: riseIn 0.2s var(--ease);
  }

  #pluginsListView, #pluginEditorView {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .plugins-toolbar { padding: 18px; border-bottom: 1px solid var(--line-soft); }

  .plugins-grid {
    padding: 18px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 12px;
    flex: 1;
  }

  .plugin-card {
    background: var(--charcoal-raised);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius-sm);
    padding: 15px;
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.12s ease;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .plugin-card:hover { border-color: var(--amber-dim); transform: translateY(-1px); }

  .plugin-card-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .plugin-card.builtin {
    border-color: var(--amber-dim);
    background: var(--amber-glow);
  }

  .builtin-badge {
    font-size: 9.5px;
    font-weight: 600;
    background: var(--amber);
    color: var(--charcoal-deep);
    padding: 2px 7px;
    border-radius: 999px;
    letter-spacing: 0.02em;
  }

  .plugin-card-author, .plugin-card-date {
    font-size: 11px;
    color: var(--chalk-dim);
  }

  .plugin-card-actions {
    display: flex;
    gap: 6px;
    margin-top: 8px;
  }

  .plugin-card-actions button {
    background: none;
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: 7px;
    padding: 5px 9px;
    font-size: 11px;
    cursor: pointer;
    transition: border-color 0.15s ease;
  }

  .plugin-card-actions button:hover { border-color: var(--amber-dim); }

  .plugin-editor-toolbar {
    display: flex;
    gap: 8px;
    padding: 14px 18px;
    border-bottom: 1px solid var(--line-soft);
    align-items: center;
  }

  .plugin-name-input {
    flex: 1;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    outline: none;
  }

  .plugin-editor-split { flex: 1; display: flex; overflow: hidden; }

  .plugin-code-textarea {
    flex: 1;
    background: var(--charcoal-deep);
    color: #D7E6DC;
    border: none;
    border-right: 1px solid var(--line-soft);
    padding: 16px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    resize: none;
    outline: none;
  }

  .plugin-preview-frame { flex: 1; border: none; background: #fff; }

  /* ---------- Plugin AI assistant panel ---------- */
  #pluginAiPanel {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 340px;
    max-width: 90vw;
    background: var(--charcoal);
    border-left: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    z-index: 20;
    box-shadow: var(--shadow-lifted);
    animation: slideInRight 0.18s var(--ease);
  }

  .plugin-editor-split { position: relative; }

  .plugin-ai-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    border-bottom: 1px solid var(--line-soft);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13.5px;
  }

  #pluginAiChat {
    flex: 1;
    overflow-y: auto;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .plugin-ai-msg {
    font-size: 13px;
    line-height: 1.5;
    max-width: 100%;
  }

  .plugin-ai-msg.user {
    align-self: flex-end;
    background: var(--amber);
    color: var(--charcoal-deep);
    padding: 9px 12px;
    border-radius: 12px 12px 3px 12px;
  }

  .plugin-ai-msg.assistant {
    background: var(--charcoal-raised);
    border: 1px solid var(--line-soft);
    padding: 10px 12px;
    border-radius: 12px 12px 12px 3px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .plugin-ai-apply-btn {
    margin-top: 8px;
    background: var(--amber);
    color: var(--charcoal-deep);
    border: none;
    border-radius: 8px;
    padding: 7px 12px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 12px;
    cursor: pointer;
  }

  .plugin-ai-hint {
    font-size: 11px;
    color: var(--chalk-dim);
    padding: 0 16px 10px;
  }

  #pluginAiForm {
    display: flex;
    gap: 8px;
    padding: 12px 16px 16px;
    border-top: 1px solid var(--line-soft);
  }

  #pluginAiInput {
    flex: 1;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: 9px;
    padding: 9px 12px;
    font-size: 13px;
    outline: none;
  }

  #pluginAiSend {
    background: var(--amber);
    color: var(--charcoal-deep);
    border: none;
    border-radius: 9px;
    padding: 0 14px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 12.5px;
    cursor: pointer;
  }

  @media (max-width: 640px) {
    .plugin-editor-split { flex-direction: column; }
    .plugin-code-textarea { border-right: none; border-bottom: 1px solid var(--line-soft); height: 40%; }
    .plugin-preview-frame { height: 60%; }
    #pluginAiPanel { width: 100%; max-width: 100%; }
  }

  /* ---------- Personas ---------- */
  #personasOverlay {
    position: fixed;
    inset: 0;
    background: rgba(10, 9, 8, 0.6);
    backdrop-filter: blur(3px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 55;
    padding: 20px;
  }

  #personasOverlay.open { display: flex; }

  #personasPanel {
    width: 100%;
    max-width: 840px;
    height: 85vh;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: var(--shadow-lifted);
    animation: riseIn 0.2s var(--ease);
  }

  #personasListView, #personaEditorView, #personaChatView {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .persona-note {
    font-size: 11.5px;
    color: var(--chalk-dim);
    margin-top: 10px;
    line-height: 1.5;
  }

  .persona-avatar {
    font-size: 22px;
    line-height: 1;
  }

  .persona-toolbar-wrap { flex-wrap: wrap; row-gap: 8px; }

  .persona-avatar-input {
    width: 52px;
    text-align: center;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: 8px;
    padding: 9px 0;
    font-size: 18px;
    outline: none;
    flex-shrink: 0;
  }

  .persona-editor-body {
    flex: 1;
    overflow-y: auto;
    padding: 18px;
    position: relative;
  }

  .persona-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: var(--chalk-dim);
    display: block;
    margin: 16px 0 6px;
  }

  .persona-label:first-child { margin-top: 0; }
  .persona-label .sub { font-weight: 400; opacity: 0.85; }

  #personaModelSelect {
    width: 100%;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 13px;
  }

  #personaPromptInput {
    width: 100%;
    box-sizing: border-box;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    line-height: 1.55;
    resize: vertical;
    outline: none;
  }

  #personaAiPanel {
    margin-top: 16px;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--charcoal-raised);
    display: flex;
    flex-direction: column;
    max-height: 320px;
  }

  #personaAiChat {
    max-height: 180px;
    overflow-y: auto;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  #personaAiForm {
    display: flex;
    gap: 8px;
    padding: 10px 14px 14px;
    border-top: 1px solid var(--line-soft);
  }

  #personaAiInput {
    flex: 1;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: 9px;
    padding: 9px 12px;
    font-size: 13px;
    outline: none;
  }

  #personaAiSend {
    background: var(--amber);
    color: var(--charcoal-deep);
    border: none;
    border-radius: 9px;
    padding: 0 14px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 12.5px;
    cursor: pointer;
  }

  .persona-chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .persona-chat-form {
    display: flex;
    gap: 10px;
    padding: 14px 18px;
    border-top: 1px solid var(--line-soft);
  }

  .persona-chat-form input {
    flex: 1;
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    font-size: 14px;
    outline: none;
  }

  .persona-chat-form button {
    background: var(--amber);
    color: var(--charcoal-deep);
    border: none;
    border-radius: var(--radius-sm);
    padding: 0 20px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    cursor: pointer;
  }

  .persona-card-provider {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    background: var(--sage-glow);
    color: var(--sage);
    padding: 2px 8px;
    border-radius: 999px;
    margin-top: 4px;
  }

  /* ------------------------------------------------------------
     Attach menu
     ------------------------------------------------------------ */
  #attachMenuOverlay { align-items: flex-end; justify-content: center; }

  #attachMenu {
    width: 100%;
    max-width: 480px;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-bottom: none;
    border-radius: 22px 22px 0 0;
    padding: 22px 18px 30px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    animation: slideUp 0.2s var(--ease);
  }

  @keyframes slideUp {
    from { transform: translateY(24px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }

  .attach-menu-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 8px;
    padding: 0 4px;
  }

  .attach-option {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--charcoal-raised);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius-sm);
    padding: 13px 15px;
    cursor: pointer;
    text-align: left;
    color: var(--chalk);
    transition: border-color 0.15s ease;
  }

  .attach-option:hover { border-color: var(--amber-dim); }

  .attach-icon { font-size: 22px; }

  .attach-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
  }

  .attach-sub { font-size: 12px; color: var(--chalk-dim); margin-top: 2px; }

  .attach-cancel {
    background: none;
    border: none;
    color: var(--chalk-dim);
    padding: 12px 0 0;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    cursor: pointer;
  }

  /* ------------------------------------------------------------
     Prompt modal
     ------------------------------------------------------------ */
  #promptModalOverlay { align-items: center; justify-content: center; padding: 20px; }

  #promptModal {
    width: 100%;
    max-width: 360px;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow-lifted);
    animation: riseIn 0.18s var(--ease);
  }

  #promptModalTitle {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 12px;
  }

  #promptModalInput {
    width: 100%;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    font-size: 14px;
    outline: none;
    margin-bottom: 14px;
  }

  #promptModalActions { display: flex; gap: 10px; }

  #promptModalActions button {
    flex: 1;
    padding: 10px 0;
    border-radius: var(--radius-sm);
    border: none;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
  }

  #promptModalGo { background: var(--amber); color: var(--charcoal-deep); }
  #promptModalCancel { background: none; border: 1px solid var(--line); color: var(--chalk); }

  /* ------------------------------------------------------------
     Auth
     ------------------------------------------------------------ */
  #authOverlay { align-items: center; justify-content: center; padding: 20px; }

  #authPanel {
    width: 100%;
    max-width: 340px;
    background: var(--charcoal);
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow-lifted);
    animation: riseIn 0.18s var(--ease);
  }

  #authBody {
    padding: 20px 22px 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  #authLoggedOutView, #authLoggedInView {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
  }

  #authBody input {
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    font-size: 14px;
    outline: none;
    width: 100%;
    box-sizing: border-box;
    display: block;
  }

  #authSubmitBtn, #authLogoutBtn {
    width: 100%;
    box-sizing: border-box;
  }

  #authError { color: var(--danger); font-size: 12px; min-height: 14px; }
  #authSubmitBtn { margin-top: 6px; }

  #authToggleModeBtn {
    background: none;
    border: none;
    color: var(--chalk-dim);
    font-size: 12.5px;
    cursor: pointer;
    padding: 4px 0;
  }

  #authWelcome { font-size: 14px; margin-bottom: 10px; }

  #authOwnerBadge {
    display: inline-block;
    background: var(--amber-glow);
    color: var(--amber);
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 999px;
    margin-bottom: 16px;
  }

  /* ------------------------------------------------------------
     Lock screen
     ------------------------------------------------------------ */
  #lockScreen {
    background: var(--charcoal);
    align-items: center;
    justify-content: center;
    backdrop-filter: none;
  }

  #lockScreen.open { display: flex; }

  #lockCard {
    background: var(--charcoal-raised);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 34px;
    width: 100%;
    max-width: 320px;
    text-align: center;
    box-shadow: var(--shadow-lifted);
  }

  #lockCard p { font-size: 13px; color: var(--chalk-dim); margin: 10px 0 22px; }

  #lockPasswordInput {
    width: 100%;
    background: var(--charcoal-deep);
    border: 1px solid var(--line);
    color: var(--chalk);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    font-size: 14px;
    outline: none;
    margin-bottom: 12px;
  }

  #lockSubmit {
    width: 100%;
    background: var(--amber);
    color: var(--charcoal-deep);
    border: none;
    border-radius: var(--radius-sm);
    padding: 12px 0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  #lockSubmit:hover { background: #eaa76a; }
  #lockError { color: var(--danger); font-size: 12px; margin-top: 10px; min-height: 14px; }

  /* ------------------------------------------------------------
     Responsive
     ------------------------------------------------------------ */
  @media (max-width: 768px) {
    #sidebarToggle { display: flex; align-items: center; justify-content: center; }

    #sidebar {
      position: fixed;
      left: 0;
      top: 0;
      height: 100vh;
      z-index: 45;
      transform: translateX(-100%);
      transition: transform 0.22s var(--ease);
      box-shadow: var(--shadow-lifted);
    }

    #sidebar.open { transform: translateX(0); }
    .sidebar-header #closeSidebar { display: flex; }
    .app { padding-top: 64px; }
  }

  @media (max-width: 480px) {
    .brand { font-size: 21px; }
    select { max-width: 108px; font-size: 12px; }
    .row { max-width: 90%; }
    #settingsPanel { width: 100%; max-width: 100%; }
    #sidebar { width: 84vw; }
  }


  /* ============================================================
     GUM 6.1 — mobile-first presentation
     Keep the full feature set, but make phones feel like an app,
     not a desktop page squeezed into a small viewport.
     ============================================================ */
  @media (max-width: 768px) {
    html, body { width: 100%; min-width: 0; overflow-x: hidden; }
    body { -webkit-text-size-adjust: 100%; }

    .layout { min-height: 100dvh; display: block; }

    #sidebarToggle {
      display: flex;
      position: fixed;
      top: max(10px, env(safe-area-inset-top));
      left: 10px;
      width: 42px;
      height: 42px;
      border-radius: 13px;
      z-index: 60;
    }

    #sidebar {
      width: min(88vw, 340px);
      height: 100dvh;
      max-height: none;
      padding: calc(12px + env(safe-area-inset-top)) 14px calc(14px + env(safe-area-inset-bottom));
      position: fixed;
      inset: 0 auto 0 0;
      z-index: 55;
      transform: translateX(-105%);
      box-shadow: 18px 0 40px rgba(0,0,0,.28);
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }

    #sidebar.open { transform: translateX(0); }
    .sidebar-header { margin-bottom: 12px; }
    .sidebar-header #closeSidebar { display: flex; }
    .side-btn.primary { min-height: 46px; font-size: 14px; }
    .quick-btn { min-height: 44px; padding: 11px 10px; font-size: 13px; }
    .saved-chat-item { min-height: 44px; }

    .app {
      width: 100%;
      max-width: none;
      height: 100dvh;
      min-height: 100dvh;
      padding: calc(10px + env(safe-area-inset-top)) 10px calc(8px + env(safe-area-inset-bottom));
      margin: 0;
    }

    header {
      min-height: 46px;
      padding: 0 0 10px 52px;
      margin-bottom: 8px;
      gap: 8px;
      position: sticky;
      top: 0;
      z-index: 20;
      background: var(--charcoal-deep);
    }

    header .brand { font-size: 22px; }
    .controls { gap: 5px; }
    .icon-btn { width: 40px; height: 40px; border-radius: 12px; }
    .controls .icon-btn { width: 38px; height: 38px; }

    #chat {
      padding: 6px 2px 12px;
      gap: 14px;
      overscroll-behavior-y: contain;
      -webkit-overflow-scrolling: touch;
    }

    .row { max-width: 92%; }
    .bubble { max-width: 100%; font-size: 15px; line-height: 1.52; }
    .row.user { align-self: flex-end; }
    .row.gum { align-self: flex-start; }

    .hint {
      display: none;
    }

    #composer {
      position: sticky;
      bottom: 0;
      z-index: 25;
      width: 100%;
      padding: 7px 0 calc(5px + env(safe-area-inset-bottom));
      gap: 6px;
      background: linear-gradient(to top, var(--charcoal-deep) 78%, transparent);
    }

    #composer textarea,
    #input {
      min-height: 44px;
      max-height: 120px;
      font-size: 16px;
      border-radius: 14px;
    }

    #composer .plus-btn,
    #composer #sendBtn { flex: 0 0 44px; width: 44px; height: 44px; }
    #modelPicker { display: none; }

    /* Turn side panels into phone-friendly full-screen sheets. */
    #settingsPanel,
    #authPanel,
    #searchPanel,
    #galleryPanel,
    #pluginsPanel,
    #personasPanel {
      width: 100%;
      max-width: 100%;
      height: 100dvh;
      max-height: 100dvh;
      border-radius: 0;
      padding-bottom: env(safe-area-inset-bottom);
    }

    .settings-body { padding: 16px; }
    .settings-header { padding: 14px 16px; }

    button, input, textarea, select { touch-action: manipulation; }
    input, textarea, select { font-size: 16px; }
  }

  @media (max-width: 420px) {
    .app { padding-left: 8px; padding-right: 8px; }
    header .brand { font-size: 20px; }
    .controls .icon-btn { width: 36px; height: 36px; }
    .row { max-width: 94%; }
    #composer { gap: 5px; }
  }

</style>
</head>
<body>

<div id="lockScreen">
  <div id="lockCard">
    <div class="brand">{{ display_name }}<span>.</span></div>
    <p>This site is password protected.</p>
    <input type="password" id="lockPasswordInput" placeholder="Enter password">
    <button id="lockSubmit">Unlock</button>
    <div id="lockError"></div>
  </div>
</div>

<button id="sidebarToggle" class="icon-btn" title="Menu">☰</button>

<div class="layout">
  <aside id="sidebar">
    <div class="sidebar-header">
      <span class="brand small">{{ display_name }}<span>.</span></span>
      <button class="icon-btn" id="closeSidebar">✕</button>
    </div>

    <button class="side-btn primary" id="newChatBtn">+ New chat</button>

    <div class="side-section-label">Quick actions</div>
    <div class="quick-grid">
      <button class="quick-btn" data-cmd="/joke">😄 Joke</button>
      <button class="quick-btn" data-cmd="/fact">💡 Fact</button>
      <button class="quick-btn" data-cmd="/quote">✒ Quote</button>
      <button class="quick-btn" data-cmd="/time">🕐 Time</button>
      <button class="quick-btn" data-cmd="/calendar">📅 Calendar</button>
      <button class="quick-btn" data-cmd="/password 16">🔑 Password</button>
      <button class="quick-btn" data-cmd="/notes">🗒 Notes</button>
      <button class="quick-btn" id="galleryBtn">🖼 Gallery</button>
      <button class="quick-btn" id="pluginsBtn">🧩 Plugins</button>
      <button class="quick-btn" id="personasBtn">🎭 Personas</button>
      <button class="quick-btn" id="exportChatBtn">⬇ Export chat</button>
    </div>

    <div class="side-section-label">Ask Gum to...</div>
    <div class="quick-grid">
      <button class="quick-btn" data-prompt-cmd="/translate" data-prompt-title="Translate what?" data-prompt-placeholder="e.g. spanish hello there">🌐 Translate</button>
      <button class="quick-btn" data-prompt-cmd="/summarize" data-prompt-title="Summarize what?" data-prompt-placeholder="Paste text to summarize">📝 Summarize</button>
      <button class="quick-btn" data-prompt-cmd="/explain" data-prompt-title="Explain what code?" data-prompt-placeholder="Paste code here">🔍 Explain code</button>
      <button class="quick-btn" data-prompt-cmd="/fix" data-prompt-title="Fix what code?" data-prompt-placeholder="Paste buggy code">🛠 Fix code</button>
      <button class="quick-btn" data-prompt-cmd="/define" data-prompt-title="Define what word?" data-prompt-placeholder="e.g. ephemeral">📖 Define</button>
      <button class="quick-btn" data-prompt-cmd="/rhyme" data-prompt-title="Rhyme with what?" data-prompt-placeholder="e.g. light">🎵 Rhyme</button>
      <button class="quick-btn" data-prompt-cmd="/calc" data-prompt-title="Calculate what?" data-prompt-placeholder="e.g. 12*7+3">🧮 Calculator</button>
      <button class="quick-btn" data-prompt-cmd="/hash" data-prompt-title="Hash what text?" data-prompt-placeholder="Text to hash">🔒 Hash</button>
    </div>

    <div class="side-section-label">Saved chats</div>
    <div id="savedChatsList" class="saved-chats-list">
      <div class="saved-chats-empty">No saved chats yet</div>
    </div>
  </aside>

  <div class="app">
    <header>
      <div class="brand">{{ display_name }}<span>.</span></div>
      <div class="controls">
        <button class="icon-btn" id="searchBtn" title="Search messages (Ctrl+K)">🔍</button>
        <button class="icon-btn account-btn" id="accountBtn" title="Account">👤</button>
        <button class="icon-btn" id="settingsBtn" title="Settings">⚙</button>
        <button class="icon-btn" id="clearBtn" title="Clear conversation">↺</button>
      </div>
    </header>

    <div id="chat"></div>
    <div class="hint">Tap the ⋯ menu on the left for quick actions, or the + to attach an image or file</div>

    <form id="composer">
      <button type="button" id="plusBtn" class="icon-btn plus-btn" title="Add">+</button>

      <div id="modelPicker">
        <button type="button" id="modelPickerBtn" title="Choose model">
          <span id="modelPickerCurrent">Gum</span>
          <span class="model-picker-caret">▾</span>
        </button>
        <div id="modelPickerMenu">
          {% for model_id, desc in models.items() %}
          <button type="button" class="model-option" data-model="{{ model_id }}" data-desc="{{ desc }}">
            <span class="model-option-name">{{ model_id.split('/')[-1] }}</span>
            <span class="model-option-desc">{{ desc }}</span>
          </button>
          {% endfor %}
        </div>
      </div>

      <select id="modelSelect" hidden>
        {% for model_id, desc in models.items() %}
        <option value="{{ model_id }}" {% if model_id == current_model %}selected{% endif %}>{{ model_id.split('/')[-1] }}</option>
        {% endfor %}
      </select>

      <input type="file" id="imageFileInput" accept="image/*" hidden>
      <input type="file" id="genericFileInput" hidden>
      <input type="text" id="userInput" placeholder="Message {{ display_name }}..." autocomplete="off">
      <button type="button" id="micBtn" class="icon-btn" title="Voice input">🎤</button>
      <button type="submit" id="sendBtn">Send</button>
    </form>
  </div>
</div>

<div id="attachMenuOverlay">
  <div id="attachMenu">
    <div class="attach-menu-title">What do you want to add?</div>
    <button class="attach-option" id="attachImageBtn">
      <span class="attach-icon">🖼</span>
      <div>
        <div class="attach-label">Add image</div>
        <div class="attach-sub">Upload a photo for {{ display_name }} to look at</div>
      </div>
    </button>
    <button class="attach-option" id="attachFileBtn">
      <span class="attach-icon">📄</span>
      <div>
        <div class="attach-label">Add file</div>
        <div class="attach-sub">Text, code, CSV, JSON, PDF — up to 2MB</div>
      </div>
    </button>
    <button class="attach-option" id="attachImagineBtn">
      <span class="attach-icon">✨</span>
      <div>
        <div class="attach-label">Generate an image</div>
        <div class="attach-sub">Describe something and {{ display_name }} will draw it</div>
      </div>
    </button>
    <button class="attach-cancel" id="attachCancelBtn">Cancel</button>
  </div>
</div>

<div id="promptModalOverlay">
  <div id="promptModal">
    <div id="promptModalTitle">Enter something</div>
    <input type="text" id="promptModalInput" placeholder="Type here...">
    <div id="promptModalActions">
      <button id="promptModalGo">Go</button>
      <button id="promptModalCancel">Cancel</button>
    </div>
  </div>
</div>

<div id="searchOverlay">
  <div id="searchPanel">
    <div class="settings-header">
      <span>Search this chat</span>
      <button class="icon-btn" id="closeSearch">✕</button>
    </div>
    <div style="padding:16px;">
      <input type="text" id="searchInput" placeholder="Search messages…" autocomplete="off">
    </div>
    <div id="searchResults"></div>
  </div>
</div>

<div id="authOverlay">
  <div id="authPanel">
    <div class="settings-header">
      <span id="authTitle">Sign in</span>
      <button class="icon-btn" id="closeAuth">✕</button>
    </div>
    <div id="authBody">
      <div id="authLoggedOutView">
        <input type="email" id="authEmailInput" placeholder="Email" autocomplete="email">
        <input type="text" id="authNameInput" placeholder="Display name (sign up only)" autocomplete="nickname" style="display:none;">
        <input type="password" id="authPasswordInput" placeholder="Password" autocomplete="current-password">
        <div id="authError"></div>
        <button id="authSubmitBtn" class="side-btn primary">Sign in</button>
        <button id="authToggleModeBtn" class="ghost small">Don't have an account? Sign up</button>
      </div>
      <div id="authLoggedInView" style="display:none;">
        <div id="authWelcome"></div>
        <div id="authOwnerBadge" style="display:none;">👑 Owner</div>
        <label style="display:block;margin:12px 0 6px;font-size:12px;opacity:.7;">Display name</label>
        <div style="display:flex;gap:8px;">
          <input type="text" id="accountDisplayNameInput" maxlength="40" autocomplete="nickname" style="flex:1;">
          <button id="accountDisplayNameBtn" class="ghost small">Save</button>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;">
          <button id="accountSessionsBtn" class="ghost small">Sessions</button>
          <button id="accountPasswordBtn" class="ghost small">Change password</button>
          <button id="accountLogoutAllBtn" class="ghost small">Sign out everywhere</button>
        </div>
        <div id="accountManageBox" style="display:none;margin:10px 0;"></div>
        <button id="authLogoutBtn" class="ghost small">Log out</button>
      </div>
    </div>
  </div>
</div>

<div id="settingsOverlay">
  <div id="settingsPanel">
    <div class="settings-header">
      <span>Settings</span>
      <button class="icon-btn" id="closeSettings">✕</button>
    </div>

    <div class="settings-body">
      <label>Theme</label>
      <div class="theme-row">
        <button class="theme-swatch dusk" data-theme="dusk" title="Dusk"></button>
        <button class="theme-swatch forest" data-theme="forest" title="Forest"></button>
        <button class="theme-swatch mono" data-theme="mono" title="Mono"></button>
      </div>

      <label>Background <span class="sub">(pick from your gallery)</span></label>
      <div class="bg-row">
        <button class="bg-swatch none active" id="bgNoneBtn" title="No background">✕</button>
        <div id="bgPickerGrid" class="bg-picker-grid"></div>
      </div>

      <div id="bubbleOpacityWrap" style="display:none;">
        <label>Bubble opacity <span id="opacityValue">0.85</span></label>
        <input type="range" id="opacitySlider" min="0.3" max="1" step="0.05" value="0.85">
      </div>

      <label>Text color</label>
      <div class="wheel-row">
        <div id="colorWheel">
          <div id="colorWheelHandle"></div>
        </div>
        <div class="wheel-info">
          <div id="colorSwatchPreview"></div>
          <button id="colorResetBtn" class="ghost small">Reset to default</button>
        </div>
      </div>

      <label>Text size</label>
      <div class="segmented" id="fontSizeGroup">
        <button data-size="small">Small</button>
        <button data-size="medium">Medium</button>
        <button data-size="large">Large</button>
      </div>

      <label>Response style</label>
      <div class="segmented" id="responseModeGroup">
        <button data-mode="thinking">🧠 Thinking</button>
        <button data-mode="instant">⚡ Instant</button>
      </div>
      <div class="range-labels">
        <span id="responseModeDescription">Takes a moment to reason through the answer — more careful, more thorough.</span>
      </div>

      <label>Creativity <span id="tempValue">1.0</span></label>
      <input type="range" id="tempSlider" min="0" max="2" step="0.1" value="1.0">
      <div class="range-labels"><span>Focused</span><span>Balanced</span><span>Wild</span></div>

      <label>Models <span class="sub">(pulled live from Groq)</span></label>
      <button id="refreshModelsBtn" class="ghost small" style="margin-top:6px;">↻ Refresh model list</button>
      <div id="refreshModelsStatus" style="font-size:11px;color:var(--chalk-dim);margin-top:6px;min-height:14px;"></div>

      <label>System prompt <span class="sub">(how {{ display_name }} behaves)</span></label>
      <textarea id="systemPromptBox" rows="6"></textarea>

      <div class="settings-actions">
        <button id="saveSettings">Save</button>
        <button id="resetSettings" class="ghost">Reset to default</button>
      </div>
    </div>
  </div>
</div>

<div id="galleryOverlay">
  <div id="galleryPanel">
    <div class="settings-header">
      <span>Image gallery</span>
      <button class="icon-btn" id="closeGallery">✕</button>
    </div>
    <div id="galleryGrid" class="gallery-grid"></div>
  </div>
</div>

<div id="pluginsOverlay">
  <div id="pluginsPanel">
    <div class="settings-header">
      <span id="pluginsHeaderTitle">Plugins</span>
      <button class="icon-btn" id="closePlugins">✕</button>
    </div>

    <div id="pluginsListView">
      <div class="plugins-toolbar">
        <button class="side-btn primary" id="makePluginBtn">+ Make plugin</button>
      </div>
      <div id="pluginsGrid" class="plugins-grid"></div>
    </div>

    <div id="pluginEditorView" style="display:none;">
      <div class="plugin-editor-toolbar">
        <input type="text" id="pluginNameInput" placeholder="Plugin name" class="plugin-name-input">
        <button id="askAiBtn" class="ghost small">✨ Ask Gum</button>
        <button id="runPluginBtn" class="ghost small">▶ Run</button>
        <button id="savePluginBtn" class="ghost small">💾 Save</button>
        <button id="backToPluginsBtn" class="ghost small">← Back</button>
      </div>
      <div class="plugin-editor-split">
        <textarea id="pluginCodeInput" class="plugin-code-textarea" spellcheck="false" placeholder="&lt;!-- Write HTML, CSS (in &lt;style&gt;), and JS (in &lt;script&gt;) here.
This runs sandboxed in your own browser only — nothing touches the server. --&gt;

&lt;h2&gt;My Plugin&lt;/h2&gt;
&lt;button onclick=&quot;alert('hi')&quot;&gt;Click me&lt;/button&gt;"></textarea>
        <iframe id="pluginPreviewFrame" class="plugin-preview-frame" sandbox="allow-scripts"></iframe>
        <div id="pluginAiPanel" style="display:none;">
          <div class="plugin-ai-header">
            <span>✨ Ask Gum to help</span>
            <button class="icon-btn" id="closePluginAi">✕</button>
          </div>
          <div id="pluginAiChat"></div>
          <div class="plugin-ai-hint">Gum can see your current code and will suggest a full replacement, or answer questions about it.</div>
          <form id="pluginAiForm">
            <input type="text" id="pluginAiInput" placeholder="e.g. 'make the button bigger' or 'add a reset button'" autocomplete="off">
            <button type="submit" id="pluginAiSend">Send</button>
          </form>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="personasOverlay">
  <div id="personasPanel">
    <div class="settings-header">
      <span id="personasHeaderTitle">Personas</span>
      <button class="icon-btn" id="closePersonas">✕</button>
    </div>

    <div id="personasListView">
      <div class="plugins-toolbar">
        <button class="side-btn primary" id="makePersonaBtn">+ Make a persona</button>
        <div class="persona-note">A persona is a custom character — a name, personality, and model choice layered on top of a real AI provider. Not a new trained model.</div>
      </div>
      <div id="personasGrid" class="plugins-grid"></div>
    </div>

    <div id="personaEditorView" style="display:none;">
      <div class="plugin-editor-toolbar persona-toolbar-wrap">
        <input type="text" id="personaAvatarInput" placeholder="🤖" maxlength="4" class="persona-avatar-input">
        <input type="text" id="personaNameInput" placeholder="Persona name" class="plugin-name-input">
        <button id="askPersonaAiBtn" class="ghost small">✨ Ask Gum for help</button>
        <button id="savePersonaBtn" class="ghost small">💾 Save</button>
        <button id="backToPersonasBtn" class="ghost small">← Back</button>
      </div>

      <div class="persona-editor-body">
        <label class="persona-label">Provider</label>
        <div class="segmented" id="personaProviderGroup"></div>

        <label class="persona-label">Model</label>
        <select id="personaModelSelect"></select>

        <label class="persona-label">Personality / instructions <span class="sub">(this is the system prompt — describe who they are, how they talk, what they know)</span></label>
        <textarea id="personaPromptInput" rows="8" placeholder="e.g. You are Captain Byte, a cheerful retro-computing enthusiast who explains programming concepts using 1980s arcade game analogies. Keep answers upbeat and under 4 sentences."></textarea>

        <div id="personaAiPanel" style="display:none;">
          <div class="plugin-ai-header">
            <span>✨ Ask Gum to help write this persona</span>
            <button class="icon-btn" id="closePersonaAi">✕</button>
          </div>
          <div id="personaAiChat"></div>
          <form id="personaAiForm">
            <input type="text" id="personaAiInput" placeholder="e.g. 'make it a grumpy pirate chef'" autocomplete="off">
            <button type="submit" id="personaAiSend">Send</button>
          </form>
        </div>

        <button class="side-btn primary" id="tryPersonaBtn" style="margin-top:18px;">💬 Try this persona</button>
      </div>
    </div>

    <div id="personaChatView" style="display:none;">
      <div class="plugin-editor-toolbar">
        <span id="personaChatTitle" style="font-family:'Space Grotesk',sans-serif;font-weight:600;"></span>
        <button id="clearPersonaChatBtn" class="ghost small">↺ Clear</button>
        <button id="backFromPersonaChatBtn" class="ghost small">← Back</button>
      </div>
      <div id="personaChatMessages" class="persona-chat-messages"></div>
      <form id="personaChatForm" class="persona-chat-form">
        <input type="text" id="personaChatInput" placeholder="Say something…" autocomplete="off">
        <button type="submit">Send</button>
      </form>
    </div>
  </div>
</div>

<script>
  let sessionId = Math.random().toString(36).substring(2, 10);
  const chatEl = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const modelSelect = document.getElementById("modelSelect");
  const clearBtn = document.getElementById("clearBtn");
  const displayName = "{{ display_name }}";

  // ---------- Model picker (custom dropdown, kept in sync with hidden #modelSelect) ----------
  const modelPickerBtn = document.getElementById("modelPickerBtn");
  const modelPickerCurrent = document.getElementById("modelPickerCurrent");
  const modelPickerMenu = document.getElementById("modelPickerMenu");

  function refreshModelOptionHighlight() {
    modelPickerMenu.querySelectorAll(".model-option").forEach(opt => {
      opt.classList.toggle("active", opt.dataset.model === modelSelect.value);
    });
  }

  function setModel(modelId) {
    modelSelect.value = modelId;
    const shortName = modelId.split("/").pop();
    modelPickerCurrent.textContent = shortName;
    modelPickerCurrent.title = modelId;
    refreshModelOptionHighlight();
  }

  function openModelPicker() {
    modelPickerMenu.classList.add("open");
    modelPickerBtn.classList.add("open");
    refreshModelOptionHighlight();
  }

  function closeModelPicker() {
    modelPickerMenu.classList.remove("open");
    modelPickerBtn.classList.remove("open");
  }

  modelPickerBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (modelPickerMenu.classList.contains("open")) closeModelPicker();
    else openModelPicker();
  });

  modelPickerMenu.addEventListener("click", (e) => {
    const opt = e.target.closest(".model-option");
    if (!opt) return;
    setModel(opt.dataset.model);
    closeModelPicker();
  });

  document.addEventListener("click", (e) => {
    if (!modelPickerBtn.contains(e.target) && !modelPickerMenu.contains(e.target)) {
      closeModelPicker();
    }
  });

  // initialize the visible picker from whatever the hidden select starts with
  setModel(modelSelect.value);

  // ---------- Auth state ----------
  // Authentication is carried by the server-set HttpOnly cookie.
  let currentUser = null; // { email, display_name, is_owner }

  function gumCsrfToken() {
    const match = document.cookie.match(/(?:^|; )gum_csrf=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  }

  async function gumFetch(url, options = {}) {
    const opts = { credentials: "same-origin", ...options };
    opts.headers = { ...(options.headers || {}) };
    if (!opts.headers["Content-Type"] && opts.body && typeof opts.body === "object") {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.body);
    }
    if (!["GET", "HEAD", "OPTIONS"].includes((opts.method || "GET").toUpperCase())) {
      const csrf = gumCsrfToken();
      if (csrf) opts.headers["X-Gum-CSRF"] = csrf;
    }
    return fetch(url, opts);
  }

  async function refreshAuthState() {
    try {
      const res = await gumFetch("/me", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin"
      });
      const data = await res.json();
      if (data.logged_in) {
        currentUser = { email: data.email, display_name: data.display_name, is_owner: data.is_owner };
      } else {
        currentUser = null;
      }
    } catch (err) {
      // network hiccup — keep existing state
    }
    updateAccountUI();
  }

  function updateAccountUI() {
    const btn = document.getElementById("accountBtn");
    btn.classList.toggle("logged-in", !!currentUser);
    btn.title = currentUser ? `Signed in as ${currentUser.display_name}` : "Sign in / Sign up";
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function formatMessage(text) {
    if (text.startsWith("__IMAGE__")) {
      const rest = text.slice("__IMAGE__".length);
      const [url, caption] = rest.split("__CAPTION__");
      return `<img class="chat-image" src="${url}" alt="${escapeHtml(caption || '')}"><div style="margin-top:6px;font-size:12px;opacity:0.6;">${escapeHtml(caption || '')}</div>`;
    }
    let escaped = escapeHtml(text);
    escaped = escaped.replace(/```([\\s\\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
    escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
    escaped = escaped.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
    return escaped;
  }

  function addMessage(text, sender) {
    // "system" messages now render identically to Gum's own replies —
    // no separate System label/styling, per user preference.
    const displaySender = sender === "system" ? "gum" : sender;

    const row = document.createElement("div");
    row.className = "row " + displaySender;

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = sender === "user" ? "You" : displayName;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatMessage(text);

    row.appendChild(label);
    row.appendChild(bubble);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
    bubble.parentRow = row;
    return bubble;
  }

  function addFileOffer(bubble, fileOffer) {
    if (!fileOffer || !bubble || !bubble.parentRow) return;
    const link = document.createElement("a");
    link.className = "file-offer";
    link.href = fileOffer.url;
    link.download = "";
    link.innerHTML = `📄 Download .${fileOffer.extension} file`;
    bubble.parentRow.appendChild(link);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function addThinking() {
    const row = document.createElement("div");
    row.className = "row gum";
    row.id = "thinkingRow";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = displayName;
    const bubble = document.createElement("div");
    bubble.className = "bubble thinking";
    bubble.innerHTML = "<span></span><span></span><span></span>";
    row.appendChild(label);
    row.appendChild(bubble);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function removeThinking() {
    const el = document.getElementById("thinkingRow");
    if (el) el.remove();
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    const isSlashCommand = message.startsWith("/");
    if (!isSlashCommand) {
      addMessage(message, "user");
    }
    input.value = "";
    sendBtn.disabled = true;
    addThinking();

    try {
      const res = await gumFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId, model: modelSelect.value })
      });
      const data = await res.json();
      removeThinking();

      if (res.status === 401) {
        currentUser = null;
        updateAccountUI();
        renderAuthModal();
        authOverlay.classList.add("open");
        addMessage("Please sign in first, then send your message again.", "gum");
        return;
      }

      if (data.cleared) {
        chatEl.innerHTML = "";
        addMessage("Conversation memory cleared.", "gum");
      } else {
        const bubble = addMessage(data.reply, "gum");
        addFileOffer(bubble, data.file_offer);
      }
    } catch (err) {
      removeThinking();
      addMessage("Couldn't reach the server. Check your connection and try again.", "gum");
    }
    sendBtn.disabled = false;
    input.focus();
  });

  // ---------- Plus / attach menu ----------
  const plusBtn = document.getElementById("plusBtn");
  const attachMenuOverlay = document.getElementById("attachMenuOverlay");
  const attachCancelBtn = document.getElementById("attachCancelBtn");
  const attachImageBtn = document.getElementById("attachImageBtn");
  const attachFileBtn = document.getElementById("attachFileBtn");
  const attachImagineBtn = document.getElementById("attachImagineBtn");
  const imageFileInput = document.getElementById("imageFileInput");
  const genericFileInput = document.getElementById("genericFileInput");

  plusBtn.addEventListener("click", () => attachMenuOverlay.classList.add("open"));
  attachCancelBtn.addEventListener("click", () => attachMenuOverlay.classList.remove("open"));
  attachMenuOverlay.addEventListener("click", (e) => {
    if (e.target === attachMenuOverlay) attachMenuOverlay.classList.remove("open");
  });

  attachImageBtn.addEventListener("click", () => {
    attachMenuOverlay.classList.remove("open");
    imageFileInput.click();
  });

  attachFileBtn.addEventListener("click", () => {
    attachMenuOverlay.classList.remove("open");
    genericFileInput.click();
  });

  attachImagineBtn.addEventListener("click", () => {
    attachMenuOverlay.classList.remove("open");
    openPromptModal("What should Gum draw?", "e.g. a fox curled up in the snow", async (value) => {
      if (!value.trim()) return;
      addMessage(value, "user");
      addThinking();
      sendBtn.disabled = true;
      try {
        const res = await gumFetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: value, session_id: sessionId })
        });
        const data = await res.json();
        removeThinking();
        if (data.url) {
          addMessage(`__IMAGE__${data.url}__CAPTION__${data.prompt}`, "gum");
        } else {
          addMessage(data.error || "Couldn't generate that image.", "system");
        }
      } catch (err) {
        removeThinking();
        addMessage("Couldn't generate that image.", "system");
      }
      sendBtn.disabled = false;
    });
  });

  imageFileInput.addEventListener("change", async () => {
    const file = imageFileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = reader.result;
      addMessage(`__IMAGE__${dataUrl}__CAPTION__Uploaded image`, "user");
      addThinking();
      sendBtn.disabled = true;

      try {
        const res = await gumFetch("/image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: dataUrl,
            question: "What's in this image?",
            session_id: sessionId
          })
        });
        const data = await res.json();
        removeThinking();
        addMessage(data.reply, "gum");
      } catch (err) {
        removeThinking();
        addMessage("Couldn't analyze that image.", "system");
      }
      sendBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    imageFileInput.value = "";
  });

  genericFileInput.addEventListener("change", async () => {
    const file = genericFileInput.files[0];
    if (!file) return;

    if (file.size > 2 * 1024 * 1024) {
      addMessage("That file is over 2MB — try a smaller one.", "system");
      genericFileInput.value = "";
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = reader.result;
      addMessage(`📄 Uploaded: ${file.name}`, "user");
      addThinking();
      sendBtn.disabled = true;

      try {
        const res = await gumFetch("/upload-file", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file: dataUrl,
            filename: file.name,
            session_id: sessionId
          })
        });
        const data = await res.json();
        removeThinking();
        addMessage(data.reply, "gum");
      } catch (err) {
        removeThinking();
        addMessage("Couldn't process that file.", "system");
      }
      sendBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    genericFileInput.value = "";
  });

  // ---------- Prompt modal (for quick actions needing text input) ----------
  const promptModalOverlay = document.getElementById("promptModalOverlay");
  const promptModalTitle = document.getElementById("promptModalTitle");
  const promptModalInput = document.getElementById("promptModalInput");
  const promptModalGo = document.getElementById("promptModalGo");
  const promptModalCancel = document.getElementById("promptModalCancel");
  let promptModalCallback = null;

  function openPromptModal(title, placeholder, onSubmit) {
    promptModalTitle.textContent = title;
    promptModalInput.placeholder = placeholder;
    promptModalInput.value = "";
    promptModalCallback = onSubmit;
    promptModalOverlay.classList.add("open");
    setTimeout(() => promptModalInput.focus(), 50);
  }

  function closePromptModal() {
    promptModalOverlay.classList.remove("open");
    promptModalCallback = null;
  }

  promptModalGo.addEventListener("click", () => {
    const value = promptModalInput.value;
    const cb = promptModalCallback;
    closePromptModal();
    if (cb) cb(value);
  });

  promptModalInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") promptModalGo.click();
  });

  promptModalCancel.addEventListener("click", closePromptModal);
  promptModalOverlay.addEventListener("click", (e) => {
    if (e.target === promptModalOverlay) closePromptModal();
  });

  async function sendRawCommand(commandText) {
    // Quick-action commands (/joke, /hash, etc.) no longer echo the raw
    // command text as a user bubble — just show the thinking state, then the result.
    addThinking();
    sendBtn.disabled = true;
    try {
      const res = await gumFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: commandText, session_id: sessionId, model: modelSelect.value })
      });
      const data = await res.json();
      removeThinking();
      if (data.cleared) {
        chatEl.innerHTML = "";
        addMessage("Conversation memory cleared.", "gum");
      } else {
        const bubble = addMessage(data.reply, "gum");
        addFileOffer(bubble, data.file_offer);
      }
    } catch (err) {
      removeThinking();
      addMessage("Couldn't reach the server.", "gum");
    }
    sendBtn.disabled = false;
  }

  // ---------- Gallery ----------
  const galleryBtn = document.getElementById("galleryBtn");
  const galleryOverlay = document.getElementById("galleryOverlay");
  const closeGallery = document.getElementById("closeGallery");
  const galleryGrid = document.getElementById("galleryGrid");

  async function openGallery() {
    galleryOverlay.classList.add("open");
    galleryGrid.innerHTML = '<div class="gallery-empty">Loading…</div>';
    try {
      const res = await gumFetch("/gallery");
      const items = await res.json();
      if (items.length === 0) {
        galleryGrid.innerHTML = '<div class="gallery-empty">No images yet — try /imagine a cat wearing sunglasses, or upload a photo</div>';
        return;
      }
      galleryGrid.innerHTML = "";
      items.forEach(item => {
        const div = document.createElement("div");
        div.className = "gallery-item";
        div.innerHTML = `<img src="${item.url}" alt=""><div class="gallery-caption">${item.type === "generated" ? "✨ " : "📎 "}${(item.prompt || "").slice(0, 40)}</div>`;
        galleryGrid.appendChild(div);
      });
    } catch (err) {
      galleryGrid.innerHTML = '<div class="gallery-empty">Couldn\\\'t load gallery.</div>';
    }
  }

  galleryBtn.addEventListener("click", () => { openGallery(); sidebar.classList.remove("open"); });
  closeGallery.addEventListener("click", () => galleryOverlay.classList.remove("open"));
  galleryOverlay.addEventListener("click", (e) => {
    if (e.target === galleryOverlay) galleryOverlay.classList.remove("open");
  });

  // ---------- Plugins ----------
  const pluginsBtn = document.getElementById("pluginsBtn");
  const pluginsOverlay = document.getElementById("pluginsOverlay");
  const closePlugins = document.getElementById("closePlugins");
  const pluginsListView = document.getElementById("pluginsListView");
  const pluginEditorView = document.getElementById("pluginEditorView");
  const pluginsGrid = document.getElementById("pluginsGrid");
  const makePluginBtn = document.getElementById("makePluginBtn");
  const pluginNameInput = document.getElementById("pluginNameInput");
  const pluginCodeInput = document.getElementById("pluginCodeInput");
  const pluginPreviewFrame = document.getElementById("pluginPreviewFrame");
  const runPluginBtn = document.getElementById("runPluginBtn");
  const savePluginBtn = document.getElementById("savePluginBtn");
  const backToPluginsBtn = document.getElementById("backToPluginsBtn");
  const pluginsHeaderTitle = document.getElementById("pluginsHeaderTitle");

  let editingPluginId = null;

  function wrapPluginHtml(rawCode) {
    // Give it a minimal HTML shell if the user just pasted a fragment
    if (/<html[\\s>]/i.test(rawCode)) return rawCode;
    return `<!DOCTYPE html><html><head><meta charset="UTF-8"><style>body{font-family:sans-serif;padding:16px;margin:0;}</style></head><body>${rawCode}</body></html>`;
  }

  function runPluginPreview() {
    pluginPreviewFrame.srcdoc = wrapPluginHtml(pluginCodeInput.value);
  }

  async function openPluginsList() {
    pluginsOverlay.classList.add("open");
    pluginsListView.style.display = "flex";
    pluginEditorView.style.display = "none";
    pluginsHeaderTitle.textContent = "Plugins";
    pluginsGrid.innerHTML = '<div class="plugins-empty">Loading…</div>';
    try {
      const res = await gumFetch("/plugins");
      const items = await res.json();
      if (items.length === 0) {
        pluginsGrid.innerHTML = '<div class="plugins-empty">No plugins yet — hit "Make plugin" to build your first one</div>';
        return;
      }
      pluginsGrid.innerHTML = "";
      items.forEach(p => {
        const card = document.createElement("div");
        card.className = "plugin-card" + (p.is_builtin ? " builtin" : "");

        const name = document.createElement("div");
        name.className = "plugin-card-name";
        name.textContent = p.name;
        if (p.is_builtin) {
          const badge = document.createElement("span");
          badge.className = "builtin-badge";
          badge.textContent = "Built-in";
          name.appendChild(badge);
        }

        const author = document.createElement("div");
        author.className = "plugin-card-author";
        author.textContent = p.is_builtin ? "Ships with Gum" : `by ${p.author_name || "Guest"}`;

        const date = document.createElement("div");
        date.className = "plugin-card-date";
        date.textContent = p.is_builtin ? "Always available" : new Date(p.updated_at).toLocaleString();

        const actions = document.createElement("div");
        actions.className = "plugin-card-actions";

        const openBtn = document.createElement("button");
        openBtn.textContent = "Open";
        openBtn.addEventListener("click", (e) => { e.stopPropagation(); openPluginEditor(p.id); });
        actions.appendChild(openBtn);

        const forkBtn = document.createElement("button");
        forkBtn.textContent = "Fork";
        forkBtn.title = "Make your own editable copy of this plugin";
        forkBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          try {
            const res = await gumFetch(`/plugins/${p.id}`);
            const full = await res.json();
            openPluginEditor(null);
            pluginNameInput.value = `${full.name} (copy)`;
            pluginCodeInput.value = full.html_code;
            runPluginPreview();
          } catch (err) {
            alert("Couldn't fork that plugin.");
          }
        });
        actions.appendChild(forkBtn);

        // Only the original author or the owner can see/use the delete button —
        // built-ins never get one, regardless of who's logged in.
        const canDelete = !p.is_builtin && currentUser && (
          currentUser.is_owner ||
          (p.author_email && p.author_email === currentUser.email)
        );

        if (canDelete) {
          const delBtn = document.createElement("button");
          delBtn.textContent = "Delete";
          delBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const res = await gumFetch(`/plugins/${p.id}`, {
              method: "DELETE",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({})
            });
            const data = await res.json();
            if (data.error) {
              alert(data.error);
              return;
            }
            openPluginsList();
          });
          actions.appendChild(delBtn);
        }

        card.appendChild(name);
        card.appendChild(author);
        card.appendChild(date);
        card.appendChild(actions);
        card.addEventListener("click", () => openPluginEditor(p.id));
        pluginsGrid.appendChild(card);
      });
    } catch (err) {
      pluginsGrid.innerHTML = '<div class="plugins-empty">Couldn\\\'t load plugins.</div>';
    }
  }

  async function openPluginEditor(pluginId) {
    pluginsListView.style.display = "none";
    pluginEditorView.style.display = "flex";
    pluginAiPanel.style.display = "none";
    pluginAiChat.innerHTML = "";

    if (pluginId) {
      editingPluginId = pluginId;
      try {
        const res = await gumFetch(`/plugins/${pluginId}`);
        const p = await res.json();
        pluginNameInput.value = p.name;
        pluginCodeInput.value = p.html_code;

        if (p.is_builtin) {
          pluginsHeaderTitle.textContent = "Built-in plugin (view only)";
          pluginNameInput.disabled = true;
          pluginCodeInput.readOnly = true;
          savePluginBtn.style.display = "none";
          askAiBtn.style.display = "none";
        } else {
          pluginsHeaderTitle.textContent = "Edit plugin";
          pluginNameInput.disabled = false;
          pluginCodeInput.readOnly = false;
          savePluginBtn.style.display = "";
          askAiBtn.style.display = "";
        }
      } catch (err) {
        pluginNameInput.value = "";
        pluginCodeInput.value = "";
      }
    } else {
      editingPluginId = null;
      pluginsHeaderTitle.textContent = "Make plugin";
      pluginNameInput.value = "";
      pluginCodeInput.value = "";
      pluginNameInput.disabled = false;
      pluginCodeInput.readOnly = false;
      savePluginBtn.style.display = "";
      askAiBtn.style.display = "";
    }
    runPluginPreview();
  }

  makePluginBtn.addEventListener("click", () => openPluginEditor(null));
  runPluginBtn.addEventListener("click", runPluginPreview);
  backToPluginsBtn.addEventListener("click", openPluginsList);

  // ---------- Plugin AI assistant ----------
  const askAiBtn = document.getElementById("askAiBtn");
  const pluginAiPanel = document.getElementById("pluginAiPanel");
  const closePluginAi = document.getElementById("closePluginAi");
  const pluginAiChat = document.getElementById("pluginAiChat");
  const pluginAiForm = document.getElementById("pluginAiForm");
  const pluginAiInput = document.getElementById("pluginAiInput");
  const pluginAiSend = document.getElementById("pluginAiSend");

  askAiBtn.addEventListener("click", () => {
    const isOpen = pluginAiPanel.style.display !== "none";
    pluginAiPanel.style.display = isOpen ? "none" : "flex";
    if (!isOpen) pluginAiInput.focus();
  });

  closePluginAi.addEventListener("click", () => {
    pluginAiPanel.style.display = "none";
  });

  function addPluginAiMessage(text, role, newCode) {
    const msg = document.createElement("div");
    msg.className = "plugin-ai-msg " + role;
    msg.textContent = text;

    if (newCode) {
      const applyBtn = document.createElement("button");
      applyBtn.className = "plugin-ai-apply-btn";
      applyBtn.textContent = "Apply to editor";
      applyBtn.addEventListener("click", () => {
        pluginCodeInput.value = newCode;
        runPluginPreview();
        applyBtn.textContent = "Applied ✓";
        applyBtn.disabled = true;
      });
      msg.appendChild(document.createElement("br"));
      msg.appendChild(applyBtn);
    }

    pluginAiChat.appendChild(msg);
    pluginAiChat.scrollTop = pluginAiChat.scrollHeight;
  }

  pluginAiForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = pluginAiInput.value.trim();
    if (!message) return;

    addPluginAiMessage(message, "user");
    pluginAiInput.value = "";
    pluginAiSend.disabled = true;

    const thinkingMsg = document.createElement("div");
    thinkingMsg.className = "plugin-ai-msg assistant";
    thinkingMsg.textContent = "Thinking…";
    pluginAiChat.appendChild(thinkingMsg);
    pluginAiChat.scrollTop = pluginAiChat.scrollHeight;

    try {
      const res = await gumFetch("/plugin-ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          current_code: pluginCodeInput.value,
          session_id: sessionId
        })
      });
      const data = await res.json();
      thinkingMsg.remove();

      // strip the code block from the displayed text if we're offering an Apply button,
      // so the chat doesn't show the same code twice
      let displayText = data.reply;
      if (data.new_code) {
        displayText = displayText.replace(/```html\\s*\\n[\\s\\S]*?```/, "").trim();
        if (!displayText) displayText = "Here's an updated version:";
      }
      addPluginAiMessage(displayText, "assistant", data.new_code);
    } catch (err) {
      thinkingMsg.remove();
      addPluginAiMessage("Couldn't reach Gum. Try again.", "assistant");
    }
    pluginAiSend.disabled = false;
  });

  savePluginBtn.addEventListener("click", async () => {
    const name = pluginNameInput.value.trim() || "Untitled plugin";
    const html_code = pluginCodeInput.value;

    try {
      let res;
      if (editingPluginId) {
        res = await gumFetch(`/plugins/${editingPluginId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, html_code })
        });
      } else {
        res = await gumFetch("/plugins", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, html_code, session_id: sessionId })
        });
      }
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      editingPluginId = data.id;
      openPluginsList();
    } catch (err) {
      alert("Couldn't save the plugin.");
    }
  });

  pluginsBtn.addEventListener("click", () => { openPluginsList(); sidebar.classList.remove("open"); });
  closePlugins.addEventListener("click", () => pluginsOverlay.classList.remove("open"));
  pluginsOverlay.addEventListener("click", (e) => {
    if (e.target === pluginsOverlay) pluginsOverlay.classList.remove("open");
  });

  // ---------- AI Personas ----------
  const personasBtn = document.getElementById("personasBtn");
  const personasOverlay = document.getElementById("personasOverlay");
  const closePersonas = document.getElementById("closePersonas");
  const personasListView = document.getElementById("personasListView");
  const personaEditorView = document.getElementById("personaEditorView");
  const personaChatView = document.getElementById("personaChatView");
  const personasGrid = document.getElementById("personasGrid");
  const makePersonaBtn = document.getElementById("makePersonaBtn");
  const personasHeaderTitle = document.getElementById("personasHeaderTitle");
  const personaAvatarInput = document.getElementById("personaAvatarInput");
  const personaNameInput = document.getElementById("personaNameInput");
  const personaProviderGroup = document.getElementById("personaProviderGroup");
  const personaModelSelect = document.getElementById("personaModelSelect");
  const personaPromptInput = document.getElementById("personaPromptInput");
  const savePersonaBtn = document.getElementById("savePersonaBtn");
  const backToPersonasBtn = document.getElementById("backToPersonasBtn");
  const tryPersonaBtn = document.getElementById("tryPersonaBtn");

  let editingPersonaId = null;
  let availableProviders = {};
  let availableGroqModels = {};
  let availableOpenaiModels = {};

  async function loadProviderOptions() {
    try {
      const res = await gumFetch("/providers");
      const data = await res.json();
      availableProviders = data.providers;
      availableGroqModels = data.groq_models;
      availableOpenaiModels = data.openai_models;

      personaProviderGroup.innerHTML = "";
      Object.entries(availableProviders).forEach(([key, info]) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.provider = key;
        btn.textContent = info.label;
        btn.addEventListener("click", () => selectPersonaProvider(key));
        personaProviderGroup.appendChild(btn);
      });
    } catch (err) {
      personaProviderGroup.innerHTML = '<span class="plugins-empty">Couldn\\\'t load providers.</span>';
    }
  }

  function selectPersonaProvider(providerKey) {
    personaProviderGroup.querySelectorAll("button").forEach(b => {
      b.classList.toggle("active", b.dataset.provider === providerKey);
    });
    const models = providerKey === "openai" ? availableOpenaiModels : availableGroqModels;
    personaModelSelect.innerHTML = "";
    Object.entries(models).forEach(([modelId, desc]) => {
      const opt = document.createElement("option");
      opt.value = modelId;
      opt.textContent = `${modelId.split("/").pop()} — ${desc}`;
      personaModelSelect.appendChild(opt);
    });
  }

  async function openPersonasList() {
    personasOverlay.classList.add("open");
    personasListView.style.display = "flex";
    personaEditorView.style.display = "none";
    personaChatView.style.display = "none";
    personasHeaderTitle.textContent = "Personas";
    personasGrid.innerHTML = '<div class="plugins-empty">Loading…</div>';
    try {
      const res = await gumFetch("/personas");
      const items = await res.json();
      if (items.length === 0) {
        personasGrid.innerHTML = '<div class="plugins-empty">No personas yet — hit "Make a persona" to build your first one</div>';
        return;
      }
      personasGrid.innerHTML = "";
      items.forEach(p => {
        const card = document.createElement("div");
        card.className = "plugin-card";

        const name = document.createElement("div");
        name.className = "plugin-card-name";
        name.innerHTML = `<span class="persona-avatar">${p.avatar}</span> ${p.name}`;

        const provider = document.createElement("div");
        provider.innerHTML = `<span class="persona-card-provider">${p.provider === "openai" ? "OpenAI" : "Groq"} · ${p.model.split("/").pop()}</span>`;

        const author = document.createElement("div");
        author.className = "plugin-card-author";
        author.textContent = `by ${p.author_name || "Guest"}`;

        const actions = document.createElement("div");
        actions.className = "plugin-card-actions";

        const chatBtn = document.createElement("button");
        chatBtn.textContent = "Chat";
        chatBtn.addEventListener("click", (e) => { e.stopPropagation(); openPersonaChat(p.id, p.name, p.avatar); });
        actions.appendChild(chatBtn);

        const editBtn = document.createElement("button");
        editBtn.textContent = "Edit";
        editBtn.addEventListener("click", (e) => { e.stopPropagation(); openPersonaEditor(p.id); });
        actions.appendChild(editBtn);

        const canDelete = currentUser && (
          currentUser.is_owner || (p.author_email && p.author_email === currentUser.email)
        );
        if (canDelete) {
          const delBtn = document.createElement("button");
          delBtn.textContent = "Delete";
          delBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const res = await gumFetch(`/personas/${p.id}`, {
              method: "DELETE",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({})
            });
            const data = await res.json();
            if (data.error) { alert(data.error); return; }
            openPersonasList();
          });
          actions.appendChild(delBtn);
        }

        card.appendChild(name);
        card.appendChild(provider);
        card.appendChild(author);
        card.appendChild(actions);
        card.addEventListener("click", () => openPersonaChat(p.id, p.name, p.avatar));
        personasGrid.appendChild(card);
      });
    } catch (err) {
      personasGrid.innerHTML = '<div class="plugins-empty">Couldn\\\'t load personas.</div>';
    }
  }

  async function openPersonaEditor(personaId) {
    personasListView.style.display = "none";
    personaEditorView.style.display = "flex";
    personaChatView.style.display = "none";
    document.getElementById("personaAiPanel").style.display = "none";
    document.getElementById("personaAiChat").innerHTML = "";

    await loadProviderOptions();

    if (personaId) {
      editingPersonaId = personaId;
      personasHeaderTitle.textContent = "Edit persona";
      try {
        const res = await gumFetch(`/personas/${personaId}`);
        const p = await res.json();
        personaAvatarInput.value = p.avatar;
        personaNameInput.value = p.name;
        personaPromptInput.value = p.system_prompt;
        selectPersonaProvider(p.provider);
        personaModelSelect.value = p.model;
      } catch (err) {
        alert("Couldn't load that persona.");
      }
    } else {
      editingPersonaId = null;
      personasHeaderTitle.textContent = "Make a persona";
      personaAvatarInput.value = "🤖";
      personaNameInput.value = "";
      personaPromptInput.value = "";
      const firstProvider = Object.keys(availableProviders)[0];
      if (firstProvider) selectPersonaProvider(firstProvider);
    }
  }

  makePersonaBtn.addEventListener("click", () => openPersonaEditor(null));
  backToPersonasBtn.addEventListener("click", openPersonasList);

  savePersonaBtn.addEventListener("click", async () => {
    const activeProviderBtn = personaProviderGroup.querySelector("button.active");
    const payload = {
      name: personaNameInput.value.trim() || "Untitled Persona",
      avatar: personaAvatarInput.value.trim() || "🤖",
      system_prompt: personaPromptInput.value,
      provider: activeProviderBtn ? activeProviderBtn.dataset.provider : "groq",
      model: personaModelSelect.value,
    };

    try {
      let res;
      if (editingPersonaId) {
        res = await gumFetch(`/personas/${editingPersonaId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
      } else {
        res = await gumFetch("/personas", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
      }
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      editingPersonaId = data.id;
      openPersonasList();
    } catch (err) {
      alert("Couldn't save the persona.");
    }
  });

  personasBtn.addEventListener("click", () => { openPersonasList(); sidebar.classList.remove("open"); });
  closePersonas.addEventListener("click", () => personasOverlay.classList.remove("open"));
  personasOverlay.addEventListener("click", (e) => {
    if (e.target === personasOverlay) personasOverlay.classList.remove("open");
  });

  // ---------- Persona AI-assist (helps write the personality/system prompt) ----------
  const askPersonaAiBtn = document.getElementById("askPersonaAiBtn");
  const personaAiPanel = document.getElementById("personaAiPanel");
  const closePersonaAi = document.getElementById("closePersonaAi");
  const personaAiChat = document.getElementById("personaAiChat");
  const personaAiForm = document.getElementById("personaAiForm");
  const personaAiInput = document.getElementById("personaAiInput");

  askPersonaAiBtn.addEventListener("click", () => {
    const isOpen = personaAiPanel.style.display !== "none";
    personaAiPanel.style.display = isOpen ? "none" : "flex";
    if (!isOpen) personaAiInput.focus();
  });
  closePersonaAi.addEventListener("click", () => { personaAiPanel.style.display = "none"; });

  function addPersonaAiMessage(text, role, newPrompt) {
    const msg = document.createElement("div");
    msg.className = "plugin-ai-msg " + role;
    msg.textContent = text;
    if (newPrompt) {
      const applyBtn = document.createElement("button");
      applyBtn.className = "plugin-ai-apply-btn";
      applyBtn.textContent = "Use this personality";
      applyBtn.addEventListener("click", () => {
        personaPromptInput.value = newPrompt;
        applyBtn.textContent = "Applied ✓";
        applyBtn.disabled = true;
      });
      msg.appendChild(document.createElement("br"));
      msg.appendChild(applyBtn);
    }
    personaAiChat.appendChild(msg);
    personaAiChat.scrollTop = personaAiChat.scrollHeight;
  }

  personaAiForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = personaAiInput.value.trim();
    if (!message) return;
    addPersonaAiMessage(message, "user");
    personaAiInput.value = "";

    const thinkingMsg = document.createElement("div");
    thinkingMsg.className = "plugin-ai-msg assistant";
    thinkingMsg.textContent = "Thinking…";
    personaAiChat.appendChild(thinkingMsg);

    try {
      const res = await gumFetch("/plugin-ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: `Write a system prompt (personality/instructions) for an AI persona based on this request: "${message}". Current draft, if any: "${personaPromptInput.value}". Respond with ONLY the system prompt text itself, no preamble, no code block, 2-6 sentences describing who they are, how they talk, and what they know.`,
          current_code: "",
          session_id: sessionId
        })
      });
      const data = await res.json();
      thinkingMsg.remove();
      const cleaned = data.reply.replace(/```[\\s\\S]*?```/g, "").trim();
      addPersonaAiMessage(cleaned, "assistant", cleaned);
    } catch (err) {
      thinkingMsg.remove();
      addPersonaAiMessage("Couldn't reach Gum. Try again.", "assistant");
    }
  });

  // ---------- Try persona (live chat with it) ----------
  let activePersonaId = null;
  const personaChatMessages = document.getElementById("personaChatMessages");
  const personaChatForm = document.getElementById("personaChatForm");
  const personaChatInput = document.getElementById("personaChatInput");
  const personaChatTitle = document.getElementById("personaChatTitle");
  const backFromPersonaChatBtn = document.getElementById("backFromPersonaChatBtn");
  const clearPersonaChatBtn = document.getElementById("clearPersonaChatBtn");

  function openPersonaChat(personaId, name, avatar) {
    activePersonaId = personaId;
    personasListView.style.display = "none";
    personaEditorView.style.display = "none";
    personaChatView.style.display = "flex";
    personaChatTitle.textContent = `${avatar || "🤖"} ${name}`;
    personaChatMessages.innerHTML = "";
    addPersonaChatMessage(`Hey, I'm ${name}. What's up?`, "gum");
  }

  function addPersonaChatMessage(text, sender) {
    const row = document.createElement("div");
    row.className = "row " + (sender === "user" ? "user" : "gum");
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = sender === "user" ? "You" : personaChatTitle.textContent;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatMessage(text);
    row.appendChild(label);
    row.appendChild(bubble);
    personaChatMessages.appendChild(row);
    personaChatMessages.scrollTop = personaChatMessages.scrollHeight;
  }

  personaChatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = personaChatInput.value.trim();
    if (!message || !activePersonaId) return;
    addPersonaChatMessage(message, "user");
    personaChatInput.value = "";

    try {
      const res = await gumFetch(`/personas/${activePersonaId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId })
      });
      const data = await res.json();
      addPersonaChatMessage(data.reply, "gum");
    } catch (err) {
      addPersonaChatMessage("Couldn't reach the server.", "gum");
    }
  });

  clearPersonaChatBtn.addEventListener("click", async () => {
    if (!activePersonaId) return;
    await gumFetch(`/personas/${activePersonaId}/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    personaChatMessages.innerHTML = "";
  });

  backFromPersonaChatBtn.addEventListener("click", openPersonasList);
  tryPersonaBtn.addEventListener("click", () => {
    if (editingPersonaId) {
      openPersonaChat(editingPersonaId, personaNameInput.value || "Persona", personaAvatarInput.value);
    } else {
      alert("Save the persona first, then you can try it.");
    }
  });

  // ---------- Auth modal ----------
  const accountBtn = document.getElementById("accountBtn");
  const authOverlay = document.getElementById("authOverlay");
  const closeAuth = document.getElementById("closeAuth");
  const authTitle = document.getElementById("authTitle");
  const authLoggedOutView = document.getElementById("authLoggedOutView");
  const authLoggedInView = document.getElementById("authLoggedInView");
  const authEmailInput = document.getElementById("authEmailInput");
  const authNameInput = document.getElementById("authNameInput");
  const authPasswordInput = document.getElementById("authPasswordInput");
  const authError = document.getElementById("authError");
  const authSubmitBtn = document.getElementById("authSubmitBtn");
  const authToggleModeBtn = document.getElementById("authToggleModeBtn");
  const authWelcome = document.getElementById("authWelcome");
  const authOwnerBadge = document.getElementById("authOwnerBadge");
  const authLogoutBtn = document.getElementById("authLogoutBtn");
  const accountSessionsBtn = document.getElementById("accountSessionsBtn");
  const accountPasswordBtn = document.getElementById("accountPasswordBtn");
  const accountLogoutAllBtn = document.getElementById("accountLogoutAllBtn");
  const accountManageBox = document.getElementById("accountManageBox");
  const accountDisplayNameInput = document.getElementById("accountDisplayNameInput");
  const accountDisplayNameBtn = document.getElementById("accountDisplayNameBtn");

  let authMode = "signin"; // or "signup"

  function renderAuthModal() {
    authError.textContent = "";
    accountManageBox.style.display = "none";
    if (currentUser) {
      authLoggedOutView.style.display = "none";
      authLoggedInView.style.display = "block";
      authTitle.textContent = "Account";
      authWelcome.textContent = `Signed in as ${currentUser.display_name} (${currentUser.email})`;
      accountDisplayNameInput.value = currentUser.display_name || "";
      authOwnerBadge.style.display = currentUser.is_owner ? "inline-block" : "none";
    } else {
      authLoggedOutView.style.display = "flex";
      authLoggedInView.style.display = "none";
      authTitle.textContent = authMode === "signin" ? "Sign in" : "Sign up";
      authNameInput.style.display = authMode === "signup" ? "block" : "none";
      authSubmitBtn.textContent = authMode === "signin" ? "Sign in" : "Sign up";
      authToggleModeBtn.textContent = authMode === "signin"
        ? "Don't have an account? Sign up"
        : "Already have an account? Sign in";
    }
  }

  accountBtn.addEventListener("click", () => {
    renderAuthModal();
    authOverlay.classList.add("open");
  });

  closeAuth.addEventListener("click", () => authOverlay.classList.remove("open"));
  authOverlay.addEventListener("click", (e) => {
    if (e.target === authOverlay) authOverlay.classList.remove("open");
  });

  authToggleModeBtn.addEventListener("click", () => {
    authMode = authMode === "signin" ? "signup" : "signin";
    renderAuthModal();
  });

  authSubmitBtn.addEventListener("click", async () => {
    const email = authEmailInput.value.trim();
    const password = authPasswordInput.value;
    const display_name = authNameInput.value.trim();

    if (!email || !password) {
      authError.textContent = "Fill in both fields.";
      return;
    }

    const endpoint = authMode === "signin" ? "/login" : "/signup";
    try {
      const res = await gumFetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, display_name })
      });
      const data = await res.json();
      if (data.error) {
        authError.textContent = data.error;
        return;
      }
      currentUser = { email: data.email, display_name: data.display_name, is_owner: data.is_owner };
      updateAccountUI();
      renderAuthModal();
    } catch (err) {
      authError.textContent = "Something went wrong. Try again.";
    }
  });

  accountDisplayNameBtn.addEventListener("click", async () => {
    const display_name = accountDisplayNameInput.value.trim();
    if (!display_name) return;
    const res = await gumFetch("/account", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({display_name})
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Couldn't update your profile.");
      return;
    }
    currentUser.display_name = data.display_name;
    updateAccountUI();
    renderAuthModal();
  });

  async function loadAccountSessions() {
    accountManageBox.style.display = "block";
    accountManageBox.textContent = "Loading sessions…";
    try {
      const res = await gumFetch("/sessions");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Couldn't load sessions.");
      accountManageBox.innerHTML = "";
      (data.sessions || []).forEach((session, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;justify-content:space-between;gap:8px;align-items:center;margin:7px 0;font-size:12px;";
        const label = document.createElement("span");
        const created = new Date(session.created_at * 1000).toLocaleString();
        const seen = new Date(session.last_seen * 1000).toLocaleString();
        label.textContent = `${session.current ? "Current session" : "Session " + (index + 1)} · ${created} · Last active ${seen} · ${session.device}`;
        const btn = document.createElement("button");
        btn.className = "ghost small";
        btn.textContent = session.current ? "Current" : "Revoke";
        btn.disabled = !!session.current;
        btn.addEventListener("click", async () => {
          await gumFetch("/sessions/revoke", {
            method: "POST", headers: {"Content-Type":"application/json"},
            body: JSON.stringify({session_id: session.session_id})
          });
          loadAccountSessions();
        });
        row.append(label, btn);
        accountManageBox.appendChild(row);
      });
      if (!(data.sessions || []).length) accountManageBox.textContent = "No active sessions.";
    } catch (err) {
      accountManageBox.textContent = err.message || "Couldn't load sessions.";
    }
  }

  accountSessionsBtn.addEventListener("click", loadAccountSessions);

  accountPasswordBtn.addEventListener("click", async () => {
    const current_password = prompt("Current password:");
    if (current_password === null) return;
    const new_password = prompt(`New password (at least ${8} characters):`);
    if (new_password === null) return;
    const res = await gumFetch("/account/password", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({current_password, new_password})
    });
    const data = await res.json();
    alert(data.success ? data.message : (data.error || "Couldn't change password."));
  });

  accountLogoutAllBtn.addEventListener("click", async () => {
    if (!confirm("Sign out all other sessions? This browser will be signed out too.")) return;
    await gumFetch("/sessions/revoke-all", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
    currentUser = null;
    updateAccountUI();
    authOverlay.classList.remove("open");
  });

  authLogoutBtn.addEventListener("click", async () => {
    await gumFetch("/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    currentUser = null;
    updateAccountUI();
    authOverlay.classList.remove("open");
  });

  clearBtn.addEventListener("click", async () => {
    await gumFetch("/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    chatEl.innerHTML = "";
  });

  // ---------- Settings ----------
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsOverlay = document.getElementById("settingsOverlay");
  const closeSettings = document.getElementById("closeSettings");
  const themeSwatches = document.querySelectorAll(".theme-swatch");
  const fontSizeGroup = document.getElementById("fontSizeGroup");
  const tempSlider = document.getElementById("tempSlider");
  const tempValue = document.getElementById("tempValue");
  const systemPromptBox = document.getElementById("systemPromptBox");
  const saveSettingsBtn = document.getElementById("saveSettings");
  const resetSettingsBtn = document.getElementById("resetSettings");

  let currentSettings = null;
  let pendingBackgroundUrl = undefined; // undefined = not yet decided this session
  let pendingTextColor = undefined;
  let pendingResponseMode = "thinking";

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    themeSwatches.forEach(sw => sw.classList.toggle("active", sw.dataset.theme === theme));
  }

  function applyFontSize(size) {
    document.body.setAttribute("data-font", size);
    fontSizeGroup.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.size === size));
  }

  const RESPONSE_MODE_DESCRIPTIONS = {
    thinking: "Takes a moment to reason through the answer — more careful, more thorough.",
    instant: "Skips the deliberation and answers as fast and directly as possible.",
  };

  function applyResponseMode(mode) {
    const group = document.getElementById("responseModeGroup");
    group.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
    document.getElementById("responseModeDescription").textContent = RESPONSE_MODE_DESCRIPTIONS[mode] || "";
    pendingResponseMode = mode;
  }

  function applyBackground(url) {
    const chatEl2 = document.getElementById("chat");
    if (url) {
      chatEl2.style.backgroundImage = `url('${url}')`;
      document.body.classList.add("has-bg-image");
    } else {
      chatEl2.style.backgroundImage = "";
      document.body.classList.remove("has-bg-image");
    }
    document.getElementById("bubbleOpacityWrap").style.display = url ? "block" : "none";
    document.getElementById("bgNoneBtn").classList.toggle("active", !url);
    document.querySelectorAll(".bg-thumb").forEach(t => t.classList.toggle("active", t.dataset.url === url));
  }

  function applyTextColor(color) {
    const style = document.getElementById("dynamicTextColorStyle") || (() => {
      const s = document.createElement("style");
      s.id = "dynamicTextColorStyle";
      document.head.appendChild(s);
      return s;
    })();
    if (color) {
      style.textContent = `.bubble, .bubble strong, .bubble code { color: ${color} !important; }`;
      document.getElementById("colorSwatchPreview").style.background = color;
    } else {
      style.textContent = "";
      document.getElementById("colorSwatchPreview").style.background = "var(--chalk)";
    }
  }

  function applyBubbleOpacity(opacity) {
    const style = document.getElementById("dynamicOpacityStyle") || (() => {
      const s = document.createElement("style");
      s.id = "dynamicOpacityStyle";
      document.head.appendChild(s);
      return s;
    })();
    style.textContent = `
      body.has-bg-image .row.user .bubble { background: color-mix(in srgb, var(--amber) ${opacity * 100}%, transparent); }
      body.has-bg-image .row.gum .bubble { background: color-mix(in srgb, var(--sage) ${opacity * 30}%, rgba(20,20,20,${opacity * 0.6})); }
    `;
  }

  async function loadSettings() {
    const res = await gumFetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    currentSettings = await res.json();
    applySettingsToUI();
  }

  function applySettingsToUI() {
    applyTheme(currentSettings.theme);
    applyFontSize(currentSettings.font_size);
    applyResponseMode(currentSettings.response_mode || "thinking");
    tempSlider.value = currentSettings.temperature;
    tempValue.textContent = Number(currentSettings.temperature).toFixed(1);
    systemPromptBox.value = currentSettings.system_prompt;
    modelSelect.value = currentSettings.model;
    setModel(currentSettings.model);
    applyBackground(currentSettings.background_url);
    applyTextColor(currentSettings.text_color);
    applyBubbleOpacity(currentSettings.bubble_opacity ?? 0.85);
    document.getElementById("opacitySlider").value = currentSettings.bubble_opacity ?? 0.85;
    document.getElementById("opacityValue").textContent = (currentSettings.bubble_opacity ?? 0.85).toFixed(2);
    positionWheelHandleFromColor(currentSettings.text_color);
    pendingBackgroundUrl = currentSettings.background_url;
    pendingTextColor = currentSettings.text_color;
  }

  settingsBtn.addEventListener("click", () => settingsOverlay.classList.add("open"));
  closeSettings.addEventListener("click", () => settingsOverlay.classList.remove("open"));
  settingsOverlay.addEventListener("click", (e) => {
    if (e.target === settingsOverlay) settingsOverlay.classList.remove("open");
  });

  themeSwatches.forEach(sw => {
    sw.addEventListener("click", () => applyTheme(sw.dataset.theme));
  });

  fontSizeGroup.querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => applyFontSize(b.dataset.size));
  });

  document.getElementById("responseModeGroup").querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => applyResponseMode(b.dataset.mode));
  });

  tempSlider.addEventListener("input", () => {
    tempValue.textContent = Number(tempSlider.value).toFixed(1);
  });

  saveSettingsBtn.addEventListener("click", async () => {
    const payload = {
      session_id: sessionId,
      theme: document.documentElement.getAttribute("data-theme"),
      font_size: document.body.getAttribute("data-font"),
      response_mode: pendingResponseMode,
      temperature: parseFloat(tempSlider.value),
      system_prompt: systemPromptBox.value,
      model: modelSelect.value,
      background_url: pendingBackgroundUrl,
      text_color: pendingTextColor,
      bubble_opacity: parseFloat(document.getElementById("opacitySlider").value)
    };
    const res = await gumFetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    currentSettings = await res.json();
    applySettingsToUI();
    settingsOverlay.classList.remove("open");
  });

  resetSettingsBtn.addEventListener("click", async () => {
    const res = await gumFetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, reset: true })
    });
    currentSettings = await res.json();
    applySettingsToUI();
  });

  // ---------- Background picker ----------
  const bgPickerGrid = document.getElementById("bgPickerGrid");
  const bgNoneBtn = document.getElementById("bgNoneBtn");

  async function populateBackgroundPicker() {
    try {
      const res = await gumFetch("/gallery");
      const items = await res.json();
      bgPickerGrid.innerHTML = "";
      if (items.length === 0) {
        bgPickerGrid.innerHTML = '<span class="bg-picker-empty">Generate or upload an image first</span>';
        return;
      }
      items.forEach(item => {
        const img = document.createElement("img");
        img.className = "bg-thumb";
        img.src = item.url;
        img.dataset.url = item.url;
        if (item.url === pendingBackgroundUrl) img.classList.add("active");
        img.addEventListener("click", () => {
          pendingBackgroundUrl = item.url;
          applyBackground(item.url);
        });
        bgPickerGrid.appendChild(img);
      });
    } catch (err) {
      bgPickerGrid.innerHTML = '<span class="bg-picker-empty">Couldn\\\'t load gallery</span>';
    }
  }

  bgNoneBtn.addEventListener("click", () => {
    pendingBackgroundUrl = null;
    applyBackground(null);
  });

  document.getElementById("opacitySlider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById("opacityValue").textContent = val.toFixed(2);
    applyBubbleOpacity(val);
  });

  // ---------- Color wheel ----------
  const colorWheel = document.getElementById("colorWheel");
  const colorWheelHandle = document.getElementById("colorWheelHandle");
  let wheelDragging = false;

  function hsvToHex(h, s, v) {
    const c = v * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = v - c;
    let r, g, b;
    if (h < 60) [r, g, b] = [c, x, 0];
    else if (h < 120) [r, g, b] = [x, c, 0];
    else if (h < 180) [r, g, b] = [0, c, x];
    else if (h < 240) [r, g, b] = [0, x, c];
    else if (h < 300) [r, g, b] = [x, 0, c];
    else [r, g, b] = [c, 0, x];
    const toHex = (n) => Math.round((n + m) * 255).toString(16).padStart(2, "0");
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }

  function positionWheelHandleFromColor(hexColor) {
    if (!hexColor) {
      colorWheelHandle.style.left = "50%";
      colorWheelHandle.style.top = "50%";
      return;
    }
    // approximate: derive hue from hex, place handle at outer edge for that hue
    const r = parseInt(hexColor.slice(1, 3), 16) / 255;
    const g = parseInt(hexColor.slice(3, 5), 16) / 255;
    const b = parseInt(hexColor.slice(5, 7), 16) / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    let h = 0;
    if (d !== 0) {
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60;
      if (h < 0) h += 360;
    }
    const radius = 44;
    const rad = (h * Math.PI) / 180;
    const x = 60 + radius * Math.cos(rad);
    const y = 60 + radius * Math.sin(rad);
    colorWheelHandle.style.left = `${x}px`;
    colorWheelHandle.style.top = `${y}px`;
  }

  function handleWheelInteraction(clientX, clientY) {
    const rect = colorWheel.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const maxRadius = rect.width / 2;
    const innerRadius = maxRadius * 0.38; // matches the ::after cutout
    let dist = Math.sqrt(dx * dx + dy * dy);

    // Clamp the handle so it can never leave the ring (between inner cutout and outer edge)
    const clampedDist = Math.max(innerRadius + 6, Math.min(maxRadius - 4, dist));
    const angle = Math.atan2(dy, dx);
    const clampedX = Math.cos(angle) * clampedDist;
    const clampedY = Math.sin(angle) * clampedDist;

    colorWheelHandle.style.left = `${rect.width / 2 + clampedX}px`;
    colorWheelHandle.style.top = `${rect.height / 2 + clampedY}px`;

    let hue = (angle * 180) / Math.PI;
    if (hue < 0) hue += 360;
    const hex = hsvToHex(hue, 1, 1);
    pendingTextColor = hex;
    applyTextColor(hex);
  }

  colorWheel.addEventListener("pointerdown", (e) => {
    wheelDragging = true;
    colorWheel.setPointerCapture(e.pointerId);
    handleWheelInteraction(e.clientX, e.clientY);
  });

  colorWheel.addEventListener("pointermove", (e) => {
    if (!wheelDragging) return;
    handleWheelInteraction(e.clientX, e.clientY);
  });

  colorWheel.addEventListener("pointerup", () => { wheelDragging = false; });
  colorWheel.addEventListener("pointercancel", () => { wheelDragging = false; });

  document.getElementById("colorResetBtn").addEventListener("click", () => {
    pendingTextColor = null;
    applyTextColor(null);
    positionWheelHandleFromColor(null);
  });

  // refresh the background picker whenever settings panel opens
  settingsBtn.addEventListener("click", populateBackgroundPicker);

  // ---------- Live model refresh ----------
  const refreshModelsBtn = document.getElementById("refreshModelsBtn");
  const refreshModelsStatus = document.getElementById("refreshModelsStatus");

  refreshModelsBtn.addEventListener("click", async () => {
    refreshModelsStatus.textContent = "Checking Groq for the latest models…";
    refreshModelsBtn.disabled = true;
    try {
      const res = await gumFetch("/models/refresh", { method: "POST" });
      const models = await res.json();
      const previousValue = modelSelect.value;

      // rebuild the hidden native select (kept for compatibility)
      modelSelect.innerHTML = "";
      Object.keys(models).forEach(modelId => {
        const opt = document.createElement("option");
        opt.value = modelId;
        opt.textContent = modelId.split("/").pop();
        modelSelect.appendChild(opt);
      });
      if ([...modelSelect.options].some(o => o.value === previousValue)) {
        modelSelect.value = previousValue;
      }

      // rebuild the visible custom picker menu to match
      modelPickerMenu.innerHTML = "";
      Object.entries(models).forEach(([modelId, desc]) => {
        const opt = document.createElement("button");
        opt.type = "button";
        opt.className = "model-option";
        opt.dataset.model = modelId;
        opt.dataset.desc = desc;
        opt.innerHTML = `<span class="model-option-name">${modelId.split("/").pop()}</span><span class="model-option-desc">${desc}</span>`;
        modelPickerMenu.appendChild(opt);
      });
      setModel(modelSelect.value);

      refreshModelsStatus.textContent = `Updated — ${Object.keys(models).length} models available.`;
    } catch (err) {
      refreshModelsStatus.textContent = "Couldn't reach Groq to refresh the list.";
    }
    refreshModelsBtn.disabled = false;
  });

  // ---------- Sidebar ----------
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const closeSidebar = document.getElementById("closeSidebar");
  const newChatBtn = document.getElementById("newChatBtn");
  const savedChatsList = document.getElementById("savedChatsList");

  sidebarToggle.addEventListener("click", () => sidebar.classList.add("open"));
  closeSidebar.addEventListener("click", () => sidebar.classList.remove("open"));

  document.querySelectorAll(".quick-btn").forEach(btn => {
    if (btn.id === "galleryBtn") return; // handled separately below

    if (btn.dataset.cmd) {
      btn.addEventListener("click", () => {
        sendRawCommand(btn.dataset.cmd);
        sidebar.classList.remove("open");
      });
    } else if (btn.dataset.promptCmd) {
      btn.addEventListener("click", () => {
        sidebar.classList.remove("open");
        openPromptModal(
          btn.dataset.promptTitle || "Enter something",
          btn.dataset.promptPlaceholder || "Type here...",
          (value) => {
            if (!value.trim()) return;
            sendRawCommand(`${btn.dataset.promptCmd} ${value}`);
          }
        );
      });
    }
  });

  newChatBtn.addEventListener("click", () => {
    location.reload();
  });

  function formatRelativeTime(iso) {
    if (!iso) return "";
    const diffMs = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  }

  async function refreshSavedChats() {
    try {
      const res = await gumFetch("/chats");
      const chats = await res.json();
      savedChatsList.innerHTML = "";
      if (chats.length === 0) {
        savedChatsList.innerHTML = '<div class="saved-chats-empty">No saved chats yet</div>';
        return;
      }
      chats.forEach(chat => {
        const item = document.createElement("div");
        item.className = "saved-chat-item";

        const title = document.createElement("div");
        title.className = "saved-chat-title";
        title.textContent = chat.title || "Untitled";
        title.title = formatRelativeTime(chat.updated_at);

        const delBtn = document.createElement("button");
        delBtn.className = "saved-chat-delete";
        delBtn.textContent = "✕";
        delBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          await gumFetch("/chats/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: chat.session_id })
          });
          refreshSavedChats();
        });

        item.appendChild(title);
        item.appendChild(delBtn);
        item.addEventListener("click", () => loadChat(chat.session_id));
        savedChatsList.appendChild(item);
      });
    } catch (err) {
      // silent fail, sidebar just stays empty
    }
  }

  async function loadChat(targetSessionId) {
    const res = await gumFetch("/chats/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: targetSessionId })
    });
    if (!res.ok) return;
    const data = await res.json();
    chatEl.innerHTML = "";
    data.messages.forEach(m => {
      addMessage(m.content, m.role === "user" ? "user" : "gum");
    });
    window.__loadedSessionId = targetSessionId;
    sessionId = targetSessionId;
    sidebar.classList.remove("open");
  }

  // periodically refresh the saved-chats list so it reflects autosaves
  setInterval(refreshSavedChats, 8000);

  // ---------- Lock screen ----------
  const lockScreen = document.getElementById("lockScreen");
  const lockPasswordInput = document.getElementById("lockPasswordInput");
  const lockSubmit = document.getElementById("lockSubmit");
  const lockError = document.getElementById("lockError");

  async function checkAccess() {
    try {
      const res = await gumFetch("/access-status");
      const data = await res.json();
      if (data.requires_password && !data.unlocked) {
        lockScreen.classList.add("open");
        return false;
      }
      return true;
    } catch (err) {
      return true; // fail open locally rather than block the UI on a network hiccup
    }
  }

  async function tryUnlockSite() {
    const password = lockPasswordInput.value;
    const res = await gumFetch("/unlock-site", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password })
    });
    const data = await res.json();
    if (data.success) {
      lockScreen.classList.remove("open");
      lockError.textContent = "";
    } else {
      lockError.textContent = data.message || "Wrong password.";
    }
  }

  lockSubmit.addEventListener("click", tryUnlockSite);
  lockPasswordInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") tryUnlockSite();
  });

  // ---------- Message search ----------
  const searchBtn = document.getElementById("searchBtn");
  const searchOverlay = document.getElementById("searchOverlay");
  const closeSearch = document.getElementById("closeSearch");
  const searchInput = document.getElementById("searchInput");
  const searchResults = document.getElementById("searchResults");

  function openSearch() {
    searchOverlay.classList.add("open");
    searchInput.value = "";
    searchResults.innerHTML = "";
    setTimeout(() => searchInput.focus(), 50);
  }

  function closeSearchPanel() {
    searchOverlay.classList.remove("open");
  }

  function runSearch(query) {
    searchResults.innerHTML = "";
    if (!query.trim()) return;
    const lower = query.toLowerCase();
    const rows = Array.from(chatEl.querySelectorAll(".row"));
    const matches = rows.filter(row => row.textContent.toLowerCase().includes(lower));

    if (matches.length === 0) {
      searchResults.innerHTML = '<div class="search-empty">No matches in this conversation.</div>';
      return;
    }

    matches.forEach(row => {
      const bubble = row.querySelector(".bubble");
      const label = row.querySelector(".label");
      if (!bubble) return;
      const snippet = bubble.textContent.slice(0, 140);
      const item = document.createElement("div");
      item.className = "search-result-item";
      item.innerHTML = `<span class="search-result-label">${label ? label.textContent : ""}</span><span class="search-result-snippet"></span>`;
      item.querySelector(".search-result-snippet").textContent = snippet;
      item.addEventListener("click", () => {
        closeSearchPanel();
        row.scrollIntoView({ behavior: "smooth", block: "center" });
        row.classList.add("search-highlight");
        setTimeout(() => row.classList.remove("search-highlight"), 1500);
      });
      searchResults.appendChild(item);
    });
  }

  searchBtn.addEventListener("click", openSearch);
  closeSearch.addEventListener("click", closeSearchPanel);
  searchOverlay.addEventListener("click", (e) => { if (e.target === searchOverlay) closeSearchPanel(); });
  searchInput.addEventListener("input", (e) => runSearch(e.target.value));

  // ---------- Keyboard shortcuts ----------
  document.addEventListener("keydown", (e) => {
    const isTypingContext = ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);

    // Ctrl/Cmd+K — open search from anywhere
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openSearch();
      return;
    }

    // Ctrl/Cmd+Enter — send message even while focused elsewhere in the composer
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && document.activeElement === input) {
      e.preventDefault();
      form.requestSubmit();
      return;
    }

    // Escape — close whichever overlay is open, checked in a sensible priority order
    if (e.key === "Escape") {
      if (searchOverlay.classList.contains("open")) { closeSearchPanel(); return; }
      if (attachMenuOverlay.classList.contains("open")) { attachMenuOverlay.classList.remove("open"); return; }
      if (promptModalOverlay.classList.contains("open")) { closePromptModal(); return; }
      if (settingsOverlay.classList.contains("open")) { settingsOverlay.classList.remove("open"); return; }
      if (galleryOverlay.classList.contains("open")) { galleryOverlay.classList.remove("open"); return; }
      if (pluginsOverlay.classList.contains("open")) { pluginsOverlay.classList.remove("open"); return; }
      if (authOverlay.classList.contains("open")) { authOverlay.classList.remove("open"); return; }
      if (sidebar.classList.contains("open")) { sidebar.classList.remove("open"); return; }
    }
  });

  // ---------- Voice input (Web Speech API — no backend needed) ----------
  const micBtn = document.getElementById("micBtn");
  const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognitionCtor) {
    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    let isListening = false;

    micBtn.addEventListener("click", () => {
      if (isListening) {
        recognition.stop();
        return;
      }
      try {
        recognition.start();
        isListening = true;
        micBtn.classList.add("listening");
        micBtn.textContent = "🔴";
      } catch (err) {
        // recognition may already be running; ignore
      }
    });

    recognition.addEventListener("result", (e) => {
      const transcript = e.results[0][0].transcript;
      input.value = (input.value ? input.value + " " : "") + transcript;
      input.focus();
    });

    recognition.addEventListener("end", () => {
      isListening = false;
      micBtn.classList.remove("listening");
      micBtn.textContent = "🎤";
    });

    recognition.addEventListener("error", () => {
      isListening = false;
      micBtn.classList.remove("listening");
      micBtn.textContent = "🎤";
    });
  } else {
    micBtn.style.display = "none"; // browser doesn't support it — hide rather than show a dead button
  }

  // ---------- Export chat ----------
  const exportChatBtn = document.getElementById("exportChatBtn");

  exportChatBtn.addEventListener("click", () => {
    const rows = Array.from(chatEl.querySelectorAll(".row"));
    if (rows.length === 0) {
      alert("Nothing to export yet.");
      return;
    }
    let markdown = `# Chat with ${displayName}\n\n`;
    rows.forEach(row => {
      const label = row.querySelector(".label");
      const bubble = row.querySelector(".bubble");
      if (!bubble) return;
      const speaker = label ? label.textContent : "";
      markdown += `**${speaker}:** ${bubble.textContent.trim()}\n\n`;
    });
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gum-chat-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    sidebar.classList.remove("open");
  });

  checkAccess().then(() => {
    loadSettings();
    refreshSavedChats();
    refreshAuthState();
    addMessage(`Hey — I'm ${displayName}. What's on your mind?`, "gum");
    input.focus();
  });
</script>

</body>
</html>
"""


@app.route("/")
def home():
    live_models = fetch_live_models()
    return render_template_string(PAGE, display_name=DISPLAY_NAME, models=live_models, current_model=CURRENT_MODEL)


@app.route("/models/refresh", methods=["POST"])
def models_refresh():
    _, auth_error = auth_required(request.get_json(silent=True) or {})
    if auth_error:
        return auth_error
    _model_cache["fetched_at"] = 0  # force a re-fetch on next call
    models = fetch_live_models()
    return jsonify(models)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    public_session_id = data.get("session_id", "default")
    session_id = scoped_session_id(public_session_id, owner_email)
    s = get_settings(session_id)

    if request.method == "POST":
        if "model" in data and data["model"] in AVAILABLE_MODELS:
            s["model"] = data["model"]
        if "temperature" in data:
            try:
                temp = float(data["temperature"])
                s["temperature"] = max(0.0, min(2.0, temp))
            except (ValueError, TypeError):
                pass
        if "system_prompt" in data and isinstance(data["system_prompt"], str) and data["system_prompt"].strip():
            s["system_prompt"] = data["system_prompt"]
        if "theme" in data and data["theme"] in ("dusk", "forest", "mono"):
            s["theme"] = data["theme"]
        if "font_size" in data and data["font_size"] in ("small", "medium", "large"):
            s["font_size"] = data["font_size"]
        if "autoscroll" in data:
            s["autoscroll"] = bool(data["autoscroll"])
        if "background_url" in data:
            bg = data["background_url"]
            if bg is None or (isinstance(bg, str) and (bg.startswith("/static/generated/") or bg.startswith("/static/uploaded/"))):
                s["background_url"] = bg
        if "text_color" in data:
            color = data["text_color"]
            if color is None or (isinstance(color, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', color)):
                s["text_color"] = color
        if "bubble_opacity" in data:
            try:
                op = float(data["bubble_opacity"])
                s["bubble_opacity"] = max(0.3, min(1.0, op))
            except (ValueError, TypeError):
                pass
        if "response_mode" in data and data["response_mode"] in RESPONSE_MODE_PRESETS:
            s["response_mode"] = data["response_mode"]
        if data.get("reset"):
            session_settings[session_id] = DEFAULT_SETTINGS.copy()
            s = session_settings[session_id]

    return jsonify(s)


@app.route("/chat", methods=["POST"])
def chat():
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"reply": "Access locked. Enter the site password to continue.", "locked": True}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"reply": f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS} messages/hour). Try again later.", "is_command": True}), 429

    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    public_session_id = data.get("session_id", "default")
    session_id = scoped_session_id(public_session_id, owner_email)
    user_message = data.get("message", "")

    s = get_settings(session_id)
    model = data.get("model") or s["model"]

    history = get_history(session_id, owner_email, public_session_id)

    # Check if it's a slash command first
    if user_message.strip().startswith("/"):
        # /imagine goes through its own stricter rate limit since generation is heavier
        if user_message.strip().lower().startswith("/imagine "):
            if not check_rate_limit(ip, generation_log, RATE_LIMIT_MAX_GENERATIONS):
                return jsonify({"reply": f"Image generation limit reached ({RATE_LIMIT_MAX_GENERATIONS}/hour). Try again later.", "is_command": True}), 429
        result = handle_command(user_message, session_id, model, build_system_message(s)["content"], owner_email)
        if result is not None:
            if result == "__CLEAR__":
                return jsonify({"cleared": True})
            return jsonify({"reply": result, "is_command": True})
        # not a recognized command, fall through to normal chat with the raw text

    history.append({"role": "user", "content": user_message})
    sys_msg = build_system_message(s)

    try:
        explicit_subagents = subagent_requested(user_message)
        if explicit_subagents:
            requested_count, subtask = _subagent_count_and_task(user_message)
        else:
            auto_spawn, requested_count, subtask = gum_wants_subagents(
                user_message, history, model, owner_email, public_session_id, s["temperature"]
            )
            if not auto_spawn:
                requested_count = 0

        if explicit_subagents or requested_count:
            findings = run_subagents(
                subtask, owner_email, public_session_id, model, s["temperature"], requested_count
            )
            reply, lead_meta = synthesize_subagents(
                subtask, findings, owner_email, public_session_id, model, s["temperature"]
            )
            reply = reply
            history.append({"role": "assistant", "content": reply})
            autosave_chat(public_session_id, history, owner_email)
            return jsonify({
                "reply": reply,
                "is_command": False,
                "file_offer": None,
                "subagents": [{"role": x.get("role"), "reply": x.get("reply")} for x in findings],
                "subagent_count": len(findings),
                "lead_agent": lead_meta,
            })

        messages = build_bounded_messages(sys_msg, history[:-1], user_message)
        response, used_provider, used_model, used_route = ai_complete(
            messages, owner_email, public_session_id, user_message, model, s["temperature"], s
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        autosave_chat(public_session_id, history, owner_email)

        file_offer = None
        detected = detect_file_worthy_content(reply)
        if detected:
            file_url = save_generated_reply_file(detected["content"], detected["extension"], session_id, owner_email)
            file_offer = {"url": file_url, "extension": detected["extension"]}

        return jsonify({"reply": reply, "is_command": False, "file_offer": file_offer})
    except Exception as e:
        return jsonify({"reply": f"Something went wrong: {str(e)}", "is_command": True}), 500


@app.route("/subagents", methods=["POST"])
@require_auth
def subagents_endpoint():
    """Explicit API for Gum's bounded multi-agent mode."""
    data = request.get_json(silent=True) or {}
    task = str(data.get("task", "")).strip()
    if not task:
        return jsonify({"error": "task is required"}), 400
    if len(task) > 12000:
        return jsonify({"error": "task is too long"}), 400
    try:
        count = max(1, min(int(data.get("count", 3)), SUBAGENT_MAX))
    except (TypeError, ValueError):
        count = 3
    session_id = str(data.get("session_id") or secrets.token_urlsafe(12))
    auth = get_auth_from_request(data)
    owner_email = (auth or {}).get("email")
    settings = get_settings(session_id)
    findings = run_subagents(task, owner_email, session_id, settings.get("model"), settings.get("temperature", 0.7), count)
    reply, lead_meta = synthesize_subagents(task, findings, owner_email, session_id, settings.get("model"), settings.get("temperature", 0.7))
    return jsonify({"reply": reply, "subagents": findings, "subagent_count": len(findings), "lead_agent": lead_meta})


@app.route("/unlock-site", methods=["POST"])
def unlock_site():
    ip = get_client_ip()
    data = request.get_json(silent=True) or {}
    password = data.get("password", "")
    if not ACCESS_PASSWORD:
        return jsonify({"success": True})
    if password == ACCESS_PASSWORD:
        unlocked_ips.add(ip)
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Wrong password."}), 401


@app.route("/access-status", methods=["GET"])
def access_status():
    ip = get_client_ip()
    return jsonify({"requires_password": bool(ACCESS_PASSWORD), "unlocked": is_unlocked(ip)})


@app.route("/signup", methods=["POST"])
def signup():
    ip = get_client_ip()
    if not check_rate_limit(ip, request_log, 10, window=3600):  # stricter cap just for signup attempts
        return jsonify({"error": "Too many attempts. Try again later."}), 429

    data = request.get_json(silent=True) or {}
    email = data.get("email", "")
    password = data.get("password", "")
    display_name = data.get("display_name", "")

    user, error = create_user(email, password, display_name)
    if error:
        return jsonify({"error": error}), 400

    token = create_session_token(email)
    response = jsonify({
        "email": email.strip().lower(),
        "display_name": user["display_name"],
        "is_owner": is_owner(email),
    })
    return set_auth_cookie(response, token)


@app.route("/login", methods=["POST"])
def login():
    ip = get_client_ip()
    if not check_rate_limit(ip, request_log, 20, window=3600):
        return jsonify({"error": "Too many attempts. Try again later."}), 429

    data = request.get_json(silent=True) or {}
    email = data.get("email", "")
    password = data.get("password", "")

    if not verify_login(email, password):
        return jsonify({"error": "Wrong email or password."}), 401

    token = create_session_token(email)
    user = users_store[email.strip().lower()]
    response = jsonify({
        "email": email.strip().lower(),
        "display_name": user["display_name"],
        "is_owner": is_owner(email),
    })
    return set_auth_cookie(response, token)


@app.route("/logout", methods=["POST"])
def logout():
    data = request.get_json(silent=True) or {}
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME, "")
    legacy_token = data.get("auth_token", "")
    revoke_session_token(cookie_token)
    revoke_session_token(legacy_token)
    return clear_auth_cookie(jsonify({"success": True}))


@app.route("/sessions", methods=["GET"])
def sessions_list():
    owner_email, error = auth_required()
    if error:
        return error
    current_token = request.cookies.get(AUTH_COOKIE_NAME, "")
    return jsonify({"sessions": list_sessions(owner_email, current_token)})


@app.route("/sessions/revoke", methods=["POST"])
def sessions_revoke():
    owner_email, error = auth_required(request.get_json(silent=True) or {})
    if error:
        return error
    session_id = (request.get_json(silent=True) or {}).get("session_id", "")
    if not revoke_session_by_id(owner_email, session_id):
        return jsonify({"error": "Session not found or already revoked."}), 404
    current = request.cookies.get(AUTH_COOKIE_NAME, "")
    row = None
    conn = _session_db_connect()
    try:
        row = conn.execute("SELECT token_hash FROM auth_sessions WHERE session_id = ?", (session_id,)).fetchone()
    finally:
        conn.close()
    response = jsonify({"success": True})
    if row and _hash_session_token(current) == row["token_hash"]:
        clear_auth_cookie(response)
    return response


@app.route("/account", methods=["GET", "POST"])
def account():
    owner_email, error = auth_required(request.get_json(silent=True) or {})
    if error:
        return error
    user = users_store.get(owner_email, {})
    if request.method == "GET":
        return jsonify({
            "email": owner_email,
            "display_name": user.get("display_name", owner_email.split("@")[0]),
            "created_at": user.get("created_at"),
            "is_owner": is_owner(owner_email),
        })

    data = request.get_json(silent=True) or {}
    display_name = data.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        return jsonify({"error": "Display name is required."}), 400
    display_name = display_name.strip()[:40]
    users_store[owner_email]["display_name"] = display_name
    save_users(users_store)
    return jsonify({"success": True, "email": owner_email, "display_name": display_name, "is_owner": is_owner(owner_email)})


@app.route("/account/password", methods=["POST"])
def account_password():
    owner_email, error = auth_required(request.get_json(silent=True) or {})
    if error:
        return error
    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")
    if not isinstance(current_password, str) or not isinstance(new_password, str):
        return jsonify({"error": "Invalid password data."}), 400
    if not check_password_hash(users_store[owner_email]["password_hash"], current_password):
        return jsonify({"error": "Current password is incorrect."}), 400
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return jsonify({"error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters."}), 400
    if new_password == current_password:
        return jsonify({"error": "Choose a different password."}), 400
    users_store[owner_email]["password_hash"] = generate_password_hash(new_password)
    save_users(users_store)
    # Password changes invalidate every other session, but keep this browser signed in.
    current_token = request.cookies.get(AUTH_COOKIE_NAME, "")
    current_hash = _hash_session_token(current_token)
    conn = _session_db_connect()
    try:
        conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE email = ? AND token_hash != ? AND revoked_at IS NULL",
            (time.time(), owner_email, current_hash),
        )
        conn.commit()
    finally:
        conn.close()
    return jsonify({"success": True, "message": "Password changed. Other sessions were signed out."})


@app.route("/sessions/revoke-all", methods=["POST"])
def sessions_revoke_all():
    owner_email, error = auth_required(request.get_json(silent=True) or {})
    if error:
        return error
    revoke_all_sessions(owner_email)
    return clear_auth_cookie(jsonify({"success": True, "revoked": "all"}))


@app.route("/me", methods=["GET", "POST"])
def me():
    data = request.get_json(silent=True) or {}
    email = get_auth_from_request(data)
    if not email:
        return jsonify({"logged_in": False})
    user = users_store.get(email, {})
    return jsonify({
        "logged_in": True,
        "email": email,
        "display_name": user.get("display_name", email.split("@")[0]),
        "is_owner": is_owner(email),
    })


@app.route("/clear", methods=["POST"])
def clear():
    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    public_session_id = data.get("session_id", "default")
    conversations[scoped_session_id(public_session_id, owner_email)] = []
    return jsonify({"status": "cleared"})


@app.route("/chats", methods=["GET"])
def chats_list():
    owner_email, error = auth_required()
    if error:
        return error
    query = request.args.get("q", "")[:100]
    chats = list_memory_chats(owner_email, query)
    if not chats:
        # Legacy JSON files remain readable and are imported on demand.
        chats = list_saved_chats(owner_email)
    return jsonify(chats)


@app.route("/chats/load", methods=["POST"])
def chats_load():
    data = request.get_json(silent=True) or {}
    owner_email, error = auth_required(data)
    if error:
        return error
    public_session_id = data.get("session_id")
    saved = load_memory_chat(owner_email, public_session_id) or load_saved_chat(public_session_id, owner_email)
    if not saved:
        return jsonify({"error": "Not found"}), 404
    conversations[scoped_session_id(public_session_id, owner_email)] = saved["messages"]
    # Import old JSON chats into the durable store once loaded.
    if load_memory_chat(owner_email, public_session_id) is None:
        try:
            persist_chat(owner_email, public_session_id, saved["messages"], saved.get("title"))
        except Exception:
            pass
    return jsonify(saved)


@app.route("/chats/rename", methods=["POST"])
def chats_rename():
    data = request.get_json(silent=True) or {}
    owner_email, error = auth_required(data)
    if error:
        return error
    session_id = data.get("session_id")
    title = data.get("title", "")
    if not isinstance(title, str) or not title.strip():
        return jsonify({"error": "Title is required."}), 400
    if not rename_memory_chat(owner_email, session_id, title):
        saved = load_saved_chat(session_id, owner_email)
        if not saved:
            return jsonify({"error": "Chat not found."}), 404
        persist_chat(owner_email, session_id, saved.get("messages", []), title)
    return jsonify({"success": True, "title": _normalize_chat_title(title)})


@app.route("/chats/search", methods=["GET"])
def chats_search():
    owner_email, error = auth_required()
    if error:
        return error
    query = request.args.get("q", "")[:100]
    if not query.strip():
        return jsonify([])
    return jsonify(list_memory_chats(owner_email, query))


@app.route("/chats/delete", methods=["POST"])
def chats_delete():
    data = request.get_json(silent=True) or {}
    owner_email, error = auth_required(data)
    if error:
        return error
    session_id = data.get("session_id")
    deleted = delete_memory_chat(owner_email, session_id)
    if not deleted:
        deleted = delete_saved_chat(session_id, owner_email)
    conversations.pop(scoped_session_id(session_id, owner_email), None)
    return jsonify({"deleted": deleted})


@app.route("/image", methods=["POST"])
def image():
    """Accepts a base64 image + question, analyzes with the vision model, and stores the upload."""
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"reply": "Access locked. Enter the site password to continue."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"reply": f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS}/hour). Try again later."}), 429

    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    public_session_id = data.get("session_id", "default")
    session_id = scoped_session_id(public_session_id, owner_email)
    image_data_url = data.get("image")
    question = data.get("question", "What's in this image?")

    if not image_data_url:
        return jsonify({"reply": "No image provided."}), 400

    s = get_settings(session_id)
    sys_msg = build_system_message(s)

    image_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    }

    try:
        stored = save_uploaded_image(image_data_url, public_session_id, question, owner_email)
    except Exception:
        stored = None

    try:
        response = client.chat.completions.create(model=VISION_MODEL, messages=[sys_msg, image_message])
        reply = response.choices[0].message.content
        history = get_history(session_id)
        history.append({"role": "user", "content": f"[sent an image] {question}"})
        history.append({"role": "assistant", "content": reply})
        autosave_chat(public_session_id, history, owner_email)
        return jsonify({"reply": reply, "stored_url": stored["url"] if stored else None})
    except Exception as e:
        return jsonify({"reply": f"Error analyzing image: {str(e)}"}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """Generates an image from a text prompt via Pollinations.ai and stores it."""
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"error": "Access locked. Enter the site password to continue."}), 401
    if not check_rate_limit(ip, generation_log, RATE_LIMIT_MAX_GENERATIONS):
        return jsonify({"error": f"Image generation limit reached ({RATE_LIMIT_MAX_GENERATIONS}/hour). Try again later."}), 429

    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    public_session_id = data.get("session_id", "default")
    session_id = scoped_session_id(public_session_id, owner_email)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    try:
        entry = generate_image(prompt, public_session_id, owner_email)
        history = get_history(session_id)
        history.append({"role": "user", "content": f"/imagine {prompt}"})
        history.append({"role": "assistant", "content": f"__IMAGE__{entry['url']}__CAPTION__{prompt}"})
        autosave_chat(public_session_id, history, owner_email)
        return jsonify({"url": entry["url"], "prompt": prompt})
    except Exception as e:
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@app.route("/gallery", methods=["GET"])
def gallery():
    owner_email = get_auth_from_request()
    if not owner_email:
        return jsonify([])
    owner = owner_email.strip().lower()
    return jsonify([x for x in image_index if x.get("owner_email", "").strip().lower() == owner])


@app.route("/plugins", methods=["GET"])
def plugins_list():
    owner_email, auth_error = auth_required()
    if auth_error:
        return auth_error
    # Built-ins are shared; user plugins are private to their creator.
    owner = owner_email.strip().lower()
    visible = BUILTIN_PLUGINS + [p for p in plugins_store if p.get("author_email", "").strip().lower() == owner]
    lightweight = [{k: v for k, v in p.items() if k != "html_code"} for p in visible]
    return jsonify(lightweight)


@app.route("/plugins/<plugin_id>", methods=["GET"])
def plugins_get(plugin_id):
    owner_email, auth_error = auth_required()
    if auth_error:
        return auth_error
    p = find_plugin(plugin_id)
    if p and (p.get("is_builtin") or p.get("author_email", "").strip().lower() == owner_email.strip().lower() or is_owner(owner_email)):
        return jsonify(p)
    return jsonify({"error": "Not found"}), 404


@app.route("/plugins", methods=["POST"])
def plugins_create():
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"error": "Access locked."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"error": "Rate limit reached. Try again later."}), 429

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    name = data.get("name", "Untitled plugin")
    html_code = data.get("html_code", "")

    if len(html_code) > MAX_PLUGIN_CODE_SIZE:
        return jsonify({"error": f"Plugin code too large. Max {MAX_PLUGIN_CODE_SIZE // 1000}KB."}), 400

    author_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    author_name = users_store.get(author_email, {}).get("display_name") or "User"

    entry = add_plugin(name, html_code, session_id, author_email, author_name)
    return jsonify(entry)


@app.route("/plugins/<plugin_id>", methods=["PUT"])
def plugins_update(plugin_id):
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"error": "Access locked."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"error": "Rate limit reached. Try again later."}), 429

    existing = find_plugin(plugin_id)
    if not existing:
        return jsonify({"error": "Plugin not found."}), 404
    if existing.get("is_builtin"):
        return jsonify({"error": "Built-in plugins can't be edited."}), 403

    data = request.get_json(silent=True) or {}
    name = data.get("name", "Untitled plugin")
    html_code = data.get("html_code", "")

    if len(html_code) > MAX_PLUGIN_CODE_SIZE:
        return jsonify({"error": f"Plugin code too large. Max {MAX_PLUGIN_CODE_SIZE // 1000}KB."}), 400

    requester_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    if existing.get("author_email") != requester_email and not is_owner(requester_email):
        return jsonify({"error": "You can only edit your own plugins."}), 403

    updated = update_plugin(plugin_id, name, html_code)
    return jsonify(updated)


@app.route("/plugins/<plugin_id>", methods=["DELETE"])
def plugins_delete(plugin_id):
    data = request.get_json(silent=True) or {}
    existing = find_plugin(plugin_id)
    if not existing:
        return jsonify({"deleted": False}), 404
    if existing.get("is_builtin"):
        return jsonify({"error": "Built-in plugins can't be deleted."}), 403

    requester_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    if existing.get("author_email") != requester_email and not is_owner(requester_email):
        return jsonify({"error": "You can only delete your own plugins."}), 403

    ok = delete_plugin(plugin_id)
    return jsonify({"deleted": ok})


# ---------- AI Persona routes ----------
@app.route("/providers", methods=["GET"])
def providers_list():
    _, auth_error = auth_required()
    if auth_error:
        return auth_error
    """Tells the frontend which providers/models are actually usable right now."""
    return jsonify({
        "providers": {k: v for k, v in AI_PROVIDERS.items() if v["available"]},
        "openai_models": OPENAI_MODELS if AI_PROVIDERS["openai"]["available"] else {},
        "groq_models": fetch_live_models(),
    })


@app.route("/personas", methods=["GET"])
def personas_list():
    owner_email, auth_error = auth_required()
    if auth_error:
        return auth_error
    owner = owner_email.strip().lower()
    visible = [p for p in personas_store if p.get("author_email", "").strip().lower() == owner or is_owner(owner_email)]
    lightweight = [{k: v for k, v in p.items() if k != "system_prompt"} for p in visible]
    return jsonify(lightweight)


@app.route("/personas/<persona_id>", methods=["GET"])
def personas_get(persona_id):
    owner_email, auth_error = auth_required()
    if auth_error:
        return auth_error
    p = find_persona(persona_id)
    if p and (p.get("author_email", "").strip().lower() == owner_email.strip().lower() or is_owner(owner_email)):
        return jsonify(p)
    return jsonify({"error": "Not found"}), 404


@app.route("/personas", methods=["POST"])
def personas_create():
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"error": "Access locked."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"error": "Rate limit reached. Try again later."}), 429

    data = request.get_json(silent=True) or {}
    name = data.get("name", "Untitled Persona")
    avatar = data.get("avatar", "🤖")
    system_prompt = data.get("system_prompt", "")
    provider = data.get("provider", "groq")
    model = data.get("model", CURRENT_MODEL)

    if not AI_PROVIDERS.get(provider, {}).get("available"):
        return jsonify({"error": f"The '{provider}' provider isn't configured on this server."}), 400
    if len(system_prompt) > MAX_PERSONA_PROMPT_SIZE:
        return jsonify({"error": f"Personality description too long. Max {MAX_PERSONA_PROMPT_SIZE} characters."}), 400

    author_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    author_name = users_store.get(author_email, {}).get("display_name") or "User"

    entry = add_persona(name, avatar, system_prompt, provider, model, author_email, author_name)
    return jsonify(entry)


@app.route("/personas/<persona_id>", methods=["PUT"])
def personas_update(persona_id):
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"error": "Access locked."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"error": "Rate limit reached. Try again later."}), 429

    existing = find_persona(persona_id)
    if not existing:
        return jsonify({"error": "Persona not found."}), 404

    data = request.get_json(silent=True) or {}
    requester_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    if existing.get("author_email") and existing["author_email"] != requester_email and not is_owner(requester_email):
        return jsonify({"error": "You can only edit your own personas."}), 403

    name = data.get("name", existing["name"])
    avatar = data.get("avatar", existing["avatar"])
    system_prompt = data.get("system_prompt", existing["system_prompt"])
    provider = data.get("provider", existing["provider"])
    model = data.get("model", existing["model"])

    if not AI_PROVIDERS.get(provider, {}).get("available"):
        return jsonify({"error": f"The '{provider}' provider isn't configured on this server."}), 400
    if len(system_prompt) > MAX_PERSONA_PROMPT_SIZE:
        return jsonify({"error": f"Personality description too long. Max {MAX_PERSONA_PROMPT_SIZE} characters."}), 400

    updated = update_persona(persona_id, name, avatar, system_prompt, provider, model)
    return jsonify(updated)


@app.route("/personas/<persona_id>", methods=["DELETE"])
def personas_delete(persona_id):
    data = request.get_json(silent=True) or {}
    existing = find_persona(persona_id)
    if not existing:
        return jsonify({"deleted": False}), 404

    requester_email = get_auth_from_request(data)
    if existing.get("author_email") and existing["author_email"] != requester_email and not is_owner(requester_email):
        return jsonify({"error": "You can only delete your own personas."}), 403

    ok = delete_persona(persona_id)
    return jsonify({"deleted": ok})


@app.route("/personas/<persona_id>/chat", methods=["POST"])
def personas_chat(persona_id):
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"reply": "Access locked. Enter the site password to continue."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"reply": f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS}/hour). Try again later."}), 429

    persona = find_persona(persona_id)
    if not persona:
        return jsonify({"reply": "That persona doesn't exist."}), 404

    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    session_id = data.get("session_id", "default")
    user_message = data.get("message", "")
    if persona.get("author_email", "").strip().lower() != owner_email.strip().lower() and not is_owner(owner_email):
        return jsonify({"reply": "You can only use your own personas."}), 403
    history_key = f"persona:{persona_id}:{chat_owner_key(owner_email)}:{session_id}"
    history = conversations.setdefault(history_key, [])

    history.append({"role": "user", "content": user_message})
    sys_msg = {"role": "system", "content": persona["system_prompt"]}

    try:
        reply = call_ai_provider(
            persona["provider"],
            persona["model"],
            [sys_msg] + history[-MAX_HISTORY:],
        )
        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Something went wrong: {str(e)}"}), 500


@app.route("/personas/<persona_id>/clear", methods=["POST"])
def personas_clear(persona_id):
    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    persona = find_persona(persona_id)
    if not persona:
        return jsonify({"error": "Persona not found."}), 404
    if persona.get("author_email", "").strip().lower() != owner_email.strip().lower() and not is_owner(owner_email):
        return jsonify({"error": "You can only clear your own persona sessions."}), 403
    session_id = data.get("session_id", "default")
    history_key = f"persona:{persona_id}:{chat_owner_key(owner_email)}:{session_id}"
    conversations[history_key] = []
    return jsonify({"status": "cleared"})


@app.route("/static/generated/<path:filename>")
def serve_generated(filename):
    owner_email, error = auth_required()
    if error:
        return error
    entry = next((x for x in image_index if x.get("filename") == filename and x.get("type") == "generated"), None)
    if not entry or entry.get("owner_email", "").strip().lower() != owner_email.strip().lower():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(GENERATED_DIR, filename)


@app.route("/static/uploaded/<path:filename>")
def serve_uploaded(filename):
    owner_email, error = auth_required()
    if error:
        return error
    entry = next((x for x in image_index if x.get("filename") == filename and x.get("type") == "uploaded"), None)
    if not entry or entry.get("owner_email", "").strip().lower() != owner_email.strip().lower():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(UPLOADED_DIR, filename)


@app.route("/static/files/<path:filename>")
def serve_file(filename):
    owner_email, error = auth_required()
    if error:
        return error
    if not private_file_allowed(filename, owner_email):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(FILES_DIR, filename)


PLUGIN_AI_SYSTEM_PROMPT = """You are a coding assistant embedded in a plugin-maker tool. Users write
single-file HTML/CSS/JS plugins that run sandboxed in their browser (no server access, no
cookies, no external state beyond what's on the page).

When the user asks you to build or change something, respond with the COMPLETE, ready-to-run
HTML for the plugin, wrapped in a single ```html code block — nothing before or after it except
a one-sentence summary of what you did. The code must be fully self-contained: inline <style> and
<script>, no external dependencies besides plain CSS/JS. Keep it clean and working, not just a
sketch.

When the user asks a question about their code rather than requesting a change (e.g. "why isn't
this working", "what does this do"), answer conversationally without necessarily providing a full
code block — just help them understand.

If the user hasn't written any code yet and their request is vague, make a reasonable, working
starting point rather than asking clarifying questions first."""


@app.route("/plugin-ai", methods=["POST"])
def plugin_ai():
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"reply": "Access locked. Enter the site password to continue."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"reply": f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS}/hour). Try again later."}), 429

    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    user_message = data.get("message", "").strip()
    current_code = data.get("current_code", "")
    public_session_id = data.get("session_id", "default")
    session_id = scoped_session_id(public_session_id, owner_email)

    if not user_message:
        return jsonify({"reply": "Say what you'd like the plugin to do."}), 400

    context = f"Here is the plugin's current code:\n```html\n{current_code}\n```\n\n" if current_code.strip() else "The plugin is currently empty.\n\n"
    prompt = context + f"User request: {user_message}"

    s = get_settings(session_id)
    messages = [
        {"role": "system", "content": PLUGIN_AI_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(model=s["model"], messages=messages)
        reply = response.choices[0].message.content

        # pull out a code block if present, so the frontend can offer an "Apply" button
        code_match = re.search(r'```html\s*\n(.*?)```', reply, re.DOTALL)
        new_code = code_match.group(1).strip() if code_match else None

        return jsonify({"reply": reply, "new_code": new_code})
    except Exception as e:
        return jsonify({"reply": f"Something went wrong: {str(e)}", "new_code": None}), 500


@app.route("/upload-file", methods=["POST"])
def upload_file():
    ip = get_client_ip()
    if not is_unlocked(ip):
        return jsonify({"reply": "Access locked. Enter the site password to continue."}), 401
    if not check_rate_limit(ip, request_log, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"reply": f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS}/hour). Try again later."}), 429

    data = request.get_json(silent=True) or {}
    owner_email, auth_error = auth_required(data)
    if auth_error:
        return auth_error
    public_session_id = data.get("session_id", "default")
    session_id = scoped_session_id(public_session_id, owner_email)
    file_data_url = data.get("file")
    original_name = data.get("filename", "upload.txt")
    question = data.get("question", "").strip() or f"Take a look at this file: {original_name}"

    if not file_data_url:
        return jsonify({"reply": "No file provided."}), 400

    try:
        saved = save_uploaded_file(file_data_url, original_name, session_id, owner_email)
    except ValueError as e:
        return jsonify({"reply": str(e)}), 400

    s = get_settings(session_id)
    sys_msg = build_system_message(s)

    if saved["text_content"] is not None:
        prompt = f"{question}\n\nFile contents ({original_name}):\n```\n{saved['text_content']}\n```"
    else:
        prompt = f"{question}\n\n(A file named {original_name} was uploaded, but its contents couldn't be read as text — it may be a binary or unsupported format for direct reading.)"

    try:
        response = client.chat.completions.create(model=s["model"], messages=[sys_msg, {"role": "user", "content": prompt}])
        reply = response.choices[0].message.content
        history = get_history(session_id)
        history.append({"role": "user", "content": f"[uploaded file: {original_name}] {question}"})
        history.append({"role": "assistant", "content": reply})
        autosave_chat(public_session_id, history, owner_email)
        return jsonify({"reply": reply, "file_url": saved["url"]})
    except Exception as e:
        return jsonify({"reply": f"Error processing file: {str(e)}"}), 500



@app.route("/usage", methods=["GET"])
def usage():
    owner_email, error = auth_required()
    if error:
        return error
    try:
        days = int(request.args.get("days", "30"))
    except ValueError:
        days = 30
    return jsonify(usage_summary(owner_email, days))

@app.route("/health", methods=["GET"])
def health():
    """Simple uptime check — useful for hosting platforms and monitoring."""
    return jsonify({
        "status": "ok",
        "app": DISPLAY_NAME,
        "version": APP_VERSION,
        "sessions": "sqlite",
        "usage": "sqlite",
        "ai": {
            "routing": True,
            "fallback": True,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "providers": [k for k, v in AI_PROVIDERS.items() if v["available"]],
        },
        "time": datetime.datetime.now().isoformat(),
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Something went wrong on the server."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GUM_PORT", "5000")))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true" and not PUBLIC_MODE
    if PUBLIC_MODE:
        if not FLASK_SECRET_KEY:
            raise RuntimeError("GUM_PUBLIC_MODE=true requires FLASK_SECRET_KEY to be set.")
        if not OWNER_EMAIL:
            raise RuntimeError("GUM_PUBLIC_MODE=true requires OWNER_EMAIL to be set.")
    app.run(host=os.environ.get("GUM_BIND_HOST", "0.0.0.0"), port=port, debug=debug_mode)
