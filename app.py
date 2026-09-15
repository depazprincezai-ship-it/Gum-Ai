import importlib.util
from pathlib import Path
from functools import wraps
import os

_source = Path(__file__).with_name("gum_v6.4_autonomous_subagents.py")
_spec = importlib.util.spec_from_file_location("gum_runtime", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load {_source}")
_module = importlib.util.module_from_spec(_spec)

# Render automatically provides RENDER_EXTERNAL_HOSTNAME (for example,
# gum-ai.onrender.com). Pass that hostname into Gum's host allow-list before
# the source module is executed, so Render requests are accepted without
# hard-coding the deployment hostname in the repository.
_render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip()
_configured_hosts = os.environ.get("GUM_PUBLIC_HOSTS", "").strip()
if _render_host:
    os.environ["GUM_PUBLIC_HOSTS"] = ",".join(
        host for host in (_configured_hosts, _render_host) if host
    )

# gum_v6 references require_auth as a route decorator, but that decorator
# was missing from the generated source. Provide the secure adapter before
# executing the module so the app can boot without exposing protected routes.
def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        data = request.get_json(silent=True) or {}
        email, error = _module.auth_required(data)
        if error:
            return error
        return view(*args, **kwargs)
    return wrapped

from flask import request
_module.require_auth = require_auth
_spec.loader.exec_module(_module)

# The Gum source computes PUBLIC_HOSTS at import time. Ensure the Render host
# is also inserted into that already-created set after module execution.
if _render_host:
    _module.PUBLIC_HOSTS.add(_render_host.lower().split(":", 1)[0])

# ---------------------------------------------------------------------------
# Durable storage
# ---------------------------------------------------------------------------
# Render Free services have an ephemeral filesystem, so Gum's JSON/SQLite
# files are not permanent across redeploys. If DATABASE_URL is configured,
# use Postgres for accounts and user-created plugins. Without it, keep the
# existing local-file behavior so local development still works.
try:
    import gum_persistence
except ImportError:
    gum_persistence = None

if gum_persistence and gum_persistence.ENABLED:
    try:
        gum_persistence.init_store()
        durable_users = gum_persistence.load_users() or {}
        if durable_users:
            _module.users_store.clear()
            _module.users_store.update(durable_users)

        _original_create_user = _module.create_user

        def durable_create_user(email, password, display_name):
            result, error = _original_create_user(email, password, display_name)
            if error is None and result:
                gum_persistence.upsert_user(email.strip().lower(), result)
            return result, error

        _module.create_user = durable_create_user

        durable_plugins = gum_persistence.load_plugins()
        if durable_plugins is not None:
            _module.plugins_store.clear()
            _module.plugins_store.extend(durable_plugins)

        def durable_save_plugins(plugins):
            gum_persistence.save_plugins(plugins)

        _module.save_plugins = durable_save_plugins
    except Exception as exc:
        print(f"[gum] durable storage unavailable: {exc}")

# ---------------------------------------------------------------------------
# AI-visible plugin context
# ---------------------------------------------------------------------------
# Plugins are browser-sandboxed HTML apps. They cannot safely be executed by
# the model/server, but Gum can inject the installed plugin catalog into every
# AI call so the assistant knows which plugins exist and can recommend them.
_original_call_ai_provider = _module.call_ai_provider


def _plugin_ai_context():
    try:
        plugins = _module.get_all_plugins()
    except Exception:
        return ""
    if not plugins:
        return ""
    lines = [
        "GUM PLUGIN CATALOG:",
        "These browser-sandboxed plugins are installed in Gum. Recommend a relevant plugin when useful.",
    ]
    for plugin in plugins[:30]:
        name = str(plugin.get("name", "Unnamed plugin"))[:80]
        plugin_id = str(plugin.get("id", ""))[:80]
        lines.append(f"- {name} (plugin id: {plugin_id})")
    lines.append("Do not claim to have executed a plugin unless the application explicitly reports that it ran.")
    return "\n".join(lines)


def call_ai_provider_with_plugins(provider, model, messages, temperature=1.0):
    plugin_context = _plugin_ai_context()
    if plugin_context:
        messages = [dict(message) for message in messages]
        system_index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if system_index is not None:
            messages[system_index]["content"] = str(messages[system_index].get("content", "")) + "\n\n" + plugin_context
        else:
            messages.insert(0, {"role": "system", "content": plugin_context})
    return _original_call_ai_provider(provider, model, messages, temperature)

_module.call_ai_provider = call_ai_provider_with_plugins

# ---------------------------------------------------------------------------
# More permanent built-in plugins
# ---------------------------------------------------------------------------
_EXTRA_BUILTIN_PLUGINS = [
    {
        "id": "builtin-calculator",
        "name": "Calculator",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-09-15T00:00:00",
        "updated_at": "2026-09-15T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset='UTF-8'><style>body{font-family:sans-serif;background:#1a1918;color:#f2ecdc;padding:20px}input,button{box-sizing:border-box;width:100%;padding:10px;margin:5px 0;border-radius:8px;border:1px solid #444}input{background:#242220;color:#fff}button{background:#e39a5c;cursor:pointer}#out{margin-top:12px;font-size:20px}</style></head><body><h3>Calculator</h3><input id='a' placeholder='Expression, e.g. 12*(4+3)'><button onclick='go()'>Calculate</button><div id='out'></div><script>function go(){const s=document.getElementById('a').value.trim();if(!/^[0-9+\\-*/().%\\s]+$/.test(s)){out.textContent='Only basic arithmetic is allowed.';return}try{out.textContent=String(Function('"use strict";return ('+s+')')())}catch(e){out.textContent='Invalid expression.'}}</script></body></html>""",
    },
    {
        "id": "builtin-text-tools",
        "name": "Text Tools",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-09-15T00:00:00",
        "updated_at": "2026-09-15T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset='UTF-8'><style>body{font-family:sans-serif;background:#1a1918;color:#f2ecdc;padding:20px}textarea{width:100%;height:180px;box-sizing:border-box;padding:10px;background:#242220;color:#fff;border:1px solid #444;border-radius:8px}button{padding:9px 12px;margin:8px 4px 0 0;border:0;border-radius:8px;background:#e39a5c}</style></head><body><h3>Text Tools</h3><textarea id='t' placeholder='Paste text here'></textarea><div><button onclick='x(0)'>UPPERCASE</button><button onclick='x(1)'>lowercase</button><button onclick='x(2)'>Title Case</button><button onclick='x(3)'>Trim Lines</button></div><script>function x(n){let e=document.getElementById('t'),v=e.value;if(n===0)e.value=v.toUpperCase();if(n===1)e.value=v.toLowerCase();if(n===2)e.value=v.toLowerCase().replace(/\\b\\w/g,c=>c.toUpperCase());if(n===3)e.value=v.split('\\n').map(s=>s.trim()).join('\\n')}</script></body></html>""",
    },
    {
        "id": "builtin-timestamp",
        "name": "Timestamp Converter",
        "is_builtin": True,
        "author_name": "Gum",
        "author_email": None,
        "created_at": "2026-09-15T00:00:00",
        "updated_at": "2026-09-15T00:00:00",
        "html_code": """<!DOCTYPE html><html><head><meta charset='UTF-8'><style>body{font-family:sans-serif;background:#1a1918;color:#f2ecdc;padding:20px}input,button{width:100%;box-sizing:border-box;padding:10px;margin:5px 0;border-radius:8px;border:1px solid #444}input{background:#242220;color:#fff}button{background:#e39a5c;border:0;cursor:pointer}#out{margin-top:12px;word-break:break-word}</style></head><body><h3>Timestamp Converter</h3><input id='v' type='datetime-local'><button onclick='go()'>Convert to Unix timestamp</button><div id='out'></div><script>function go(){const d=new Date(v.value);out.textContent=Number.isNaN(d.getTime())?'Invalid date':Math.floor(d.getTime()/1000)}</script></body></html>""",
    },
]

_existing_ids = {p.get("id") for p in _module.BUILTIN_PLUGINS}
for _plugin in _EXTRA_BUILTIN_PLUGINS:
    if _plugin["id"] not in _existing_ids:
        _module.BUILTIN_PLUGINS.append(_plugin)

app = _module.app

# Desktop readability: keep the existing mobile sizing untouched, while
# making Gum's main chat bubbles noticeably easier to read on wider screens.
_DESKTOP_FONT_CSS = """
<style id="gum-desktop-font-size">
@media (min-width: 900px) {
  body[data-font="small"] { --base-font-size: 16px !important; }
  body[data-font="medium"] { --base-font-size: 18px !important; }
  body[data-font="large"] { --base-font-size: 20px !important; }
  .bubble { font-size: var(--base-font-size) !important; }
}
@media (min-width: 1400px) {
  body[data-font="small"] { --base-font-size: 17px !important; }
  body[data-font="medium"] { --base-font-size: 19px !important; }
  body[data-font="large"] { --base-font-size: 21px !important; }
}
</style>
"""

@app.after_request
def add_desktop_font_size(response):
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        html = response.get_data(as_text=True)
        if "id=\"gum-desktop-font-size\"" not in html and "id='gum-desktop-font-size'" not in html:
            if "</head>" in html:
                html = html.replace("</head>", _DESKTOP_FONT_CSS + "</head>", 1)
                response.set_data(html)
    return response
