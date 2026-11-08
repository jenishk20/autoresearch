"""Layer 1 of containment: AST allowlist for agent-written code (main plan §9)."""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"math", "numpy", "pandas"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals",
                   "locals", "vars", "input", "breakpoint", "memoryview", "help", "exit", "quit"}
MAX_BYTES = 4096


def check(src: str, required_func: str) -> list[str]:
    """Returns a list of problems; empty means accepted."""
    problems = []
    if len(src.encode()) > MAX_BYTES:
        problems.append(f"too long ({len(src.encode())} bytes > {MAX_BYTES})")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if required_func not in funcs:
        problems.append(f"must define {required_func}()")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"from {node.module} import not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"name {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute {node.attr} not allowed")
        elif isinstance(node, ast.ClassDef):
            problems.append("classes not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal not allowed")
        elif isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
            problems.append("while True not allowed")
    return sorted(set(problems))
