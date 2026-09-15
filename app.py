import importlib.util
from pathlib import Path
from functools import wraps
import os

_source = Path(__file__).with_name("gum_v6.4_autonomous_subagents.py")
_spec = importlib.util.spec_from_file_location("gum_runtime", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load {_source}")
_module = importlib.util.module_from_spec(_spec)

_render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip()
_configured_hosts = os.environ.get("GUM_PUBLIC_HOSTS", "").strip()
if _render_host:
    os.environ["GUM_PUBLIC_HOSTS"] = ",".join(
        host for host in (_configured_hosts, _render_host) if host
    )

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

if _render_host:
    _module.PUBLIC_HOSTS.add(_render_host.lower().split(":", 1)[0])

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

_DESKTOP_FONT_CSS = """
<style id="gum-desktop-fit">
@media (min-width: 900px) {
  :root { width:100%; height:100%; }
  html, body {
    width:100% !important;
    height:100% !important;
    min-height:100% !important;
    margin:0 !important;
    padding:0 !important;
    overflow:hidden !important;
  }
  body {
    box-sizing:border-box !important;
    font-size:18px !important;
    line-height:1.35 !important;
  }
  *, *::before, *::after { box-sizing:border-box !important; }
  button, input, textarea, select { font-size:16px !important; }
  button { min-height:42px; }
  .bubble { font-size:18px !important; line-height:1.4 !important; }
  h1 { font-size:28px !important; }
  h2 { font-size:24px !important; }
  h3 { font-size:20px !important; }

  body > * { max-height:100vh !important; }

  header, nav, .header, .topbar, .top-bar, .navbar, .toolbar {
    position:relative !important;
    z-index:100 !important;
    flex-shrink:0 !important;
  }

  button, input, textarea, select, a, [role="button"] {
    position:relative;
    z-index:101 !important;
    pointer-events:auto !important;
  }

  main, .main, .app, .app-container, .container, .chat-container {
    min-height:0 !important;
  }

  .messages, .message-list, .chat-messages, .chat-history, .conversation,
  [class*="message-list"], [class*="chat-history"], [class*="messages"] {
    min-height:0 !important;
    overflow-y:auto !important;
    overflow-x:hidden !important;
  }

  form, .composer, .chat-input, .input-area, .message-input {
    flex-shrink:0 !important;
  }

  header, nav, main, section, .header, .topbar, .toolbar {
    max-width:100% !important;
  }
}
</style>
"""

_DESKTOP_FIT_JS = """
<script id="gum-desktop-fit-js">
(function () {
  function fitDesktop() {
    if (!window.matchMedia || !window.matchMedia("(min-width: 900px)").matches) return;
    document.documentElement.classList.add("gum-desktop");
    document.body.classList.add("gum-desktop");

    var candidates = Array.from(document.body.children);
    var root = candidates.find(function (el) {
      return el && el.getBoundingClientRect && el.getBoundingClientRect().height > 0;
    });
    if (root) {
      root.style.height = "100vh";
      root.style.maxHeight = "100vh";
      root.style.minHeight = "0";
      root.style.display = "flex";
      root.style.flexDirection = "column";
      root.style.overflow = "hidden";
    }

    document.querySelectorAll('button, input, textarea, select, a, [role="button"]').forEach(function (el) {
      el.style.pointerEvents = "auto";
      if (el.tagName === "BUTTON" || el.getAttribute("role") === "button") {
        el.style.position = "relative";
        el.style.zIndex = "1000";
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", fitDesktop);
  } else {
    fitDesktop();
  }
  window.addEventListener("resize", fitDesktop);
})();
</script>
"""

@app.after_request
def add_desktop_font_size(response):
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        html = response.get_data(as_text=True)
        if "id=\"gum-desktop-fit\"" not in html and "id='gum-desktop-fit'" not in html:
            if "</head>" in html:
                html = html.replace("</head>", _DESKTOP_FONT_CSS + _DESKTOP_FIT_JS + "</head>", 1)
                response.set_data(html)
    return response
