"""Static checks on agent-written strategy code before it runs.

strategy.py receives only the point-in-time bundle and an LLM handle. To keep the evaluation
honest (no fetching of future data, no reading of results files, no tampering with the harness)
the code may import only pure-computation modules and may not touch files, the network,
introspection or dynamic code execution."""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"json", "math", "statistics", "re", "datetime", "textwrap", "collections",
                   "itertools", "functools", "typing", "dataclasses", "copy", "heapq", "bisect",
                   "engine.schema", "__future__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview", "help",
                   "exit", "quit"}
MAX_BYTES = 120_000


class GuardError(ValueError):
    pass


def check(source: str) -> None:
    if len(source.encode()) > MAX_BYTES:
        raise GuardError(f"strategy.py is larger than {MAX_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise GuardError(f"SyntaxError: {exc}") from exc
    has_analyze = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    raise GuardError(f"import {a.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS or node.level:
                raise GuardError(f"from {node.module} import ... is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise GuardError(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise GuardError(f"private attribute access ({node.attr}) is not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "analyze":
            args = [a.arg for a in node.args.args]
            if args[:3] != ["bundle", "llm", "previous"]:
                raise GuardError("analyze must be defined as analyze(bundle, llm, previous)")
            has_analyze = True
    if not has_analyze:
        raise GuardError("strategy.py must define analyze(bundle, llm, previous)")
