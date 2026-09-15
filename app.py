import importlib.util
from pathlib import Path
from functools import wraps

_source = Path(__file__).with_name("gum_v6.4_autonomous_subagents.py")
_spec = importlib.util.spec_from_file_location("gum_runtime", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load {_source}")
_module = importlib.util.module_from_spec(_spec)

# gum_v6.4 references require_auth as a route decorator, but that decorator
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
app = _module.app
