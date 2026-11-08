"""Loads agent-written code (Level 2) from /job. It runs only inside the sandbox; the control
plane's safety.py has already AST-checked it unless a red-team preset disabled that layer."""
import importlib.util

from .paths import JOB


def load_function(filename: str, func_name: str):
    path = JOB / filename
    spec = importlib.util.spec_from_file_location("agent_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise RuntimeError(f"{filename} does not define {func_name}()")
    return fn
