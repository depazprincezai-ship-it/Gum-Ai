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

# The source module imports Flask's request itself. We only need a reference
# while defining the adapter; the module will populate it during execution.
from flask import request
_module.require_auth = require_auth
_spec.loader.exec_module(_module)

# The Gum source computes PUBLIC_HOSTS at import time. Ensure the Render host
# is also inserted into that already-created set after module execution. This
# makes the allow-list work even if the source's dotenv loading or import-time
# environment handling changes in the future.
if _render_host:
    _module.PUBLIC_HOSTS.add(_render_host.lower().split(":", 1)[0])

app = _module.app
