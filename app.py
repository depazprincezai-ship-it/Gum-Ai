import importlib.util
from pathlib import Path

_source = Path(__file__).with_name("gum_v6.4_autonomous_subagents.py")
_spec = importlib.util.spec_from_file_location("gum_runtime", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
app = _module.app
