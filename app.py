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
  :root, html, body { width:100% !important; height:100% !important; min-height:100% !important; margin:0 !important; padding:0 !important; overflow:hidden !important; }
  *, *::before, *::after { box-sizing:border-box !important; }
  body { font-size:54px !important; line-height:1.15 !important; }
  body p, body span, body label, body li, body td, body th, body div, body a, body .bubble { font-size:54px !important; line-height:1.2 !important; }
  body h1 { font-size:72px !important; line-height:1.05 !important; }
  body h2 { font-size:64px !important; line-height:1.05 !important; }
  body h3, body h4 { font-size:58px !important; line-height:1.05 !important; }
  button, input, textarea, select { font-size:48px !important; line-height:1.1 !important; min-height:64px !important; }
  header, nav, .header, .topbar, .top-bar, .navbar, .toolbar { position:relative !important; z-index:9999 !important; flex-shrink:0 !important; }
  button, input, textarea, select, a, [role="button"] { position:relative !important; z-index:10000 !important; pointer-events:auto !important; }
  main, .main, .app, .app-container, .container, .chat-container { min-height:0 !important; max-height:100vh !important; }
  .messages, .message-list, .chat-messages, .chat-history, .conversation, [class*="message-list"], [class*="chat-history"], [class*="messages"] { min-height:0 !important; overflow-y:auto !important; overflow-x:hidden !important; }
  form, .composer, .chat-input, .input-area, .message-input { flex-shrink:0 !important; }
}
</style>
"""

_DESKTOP_FIT_JS = """
<script id="gum-desktop-fit-js">
(function () {
  var STYLE_ID = "gum-desktop-fit";
  var CSS = document.getElementById(STYLE_ID) ? document.getElementById(STYLE_ID).outerHTML : null;
  function enforce() {
    if (!window.matchMedia || !window.matchMedia("(min-width: 900px)").matches) return;
    document.documentElement.classList.add("gum-desktop");
    document.body.classList.add("gum-desktop");
    var style = document.getElementById(STYLE_ID);
    if (!style && CSS) {
      document.head.insertAdjacentHTML("beforeend", CSS);
    }
    document.documentElement.style.setProperty("height", "100%", "important");
    document.documentElement.style.setProperty("overflow", "hidden", "important");
    document.body.style.setProperty("height", "100%", "important");
    document.body.style.setProperty("overflow", "hidden", "important");
  }
  function start() {
    enforce();
    if (window.MutationObserver) {
      var observer = new MutationObserver(function () { enforce(); });
      observer.observe(document.documentElement, {childList:true, subtree:true, attributes:true, attributeFilter:["class","style"]});
    }
    window.addEventListener("resize", enforce, {passive:true});
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, {once:true});
  else start();
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
