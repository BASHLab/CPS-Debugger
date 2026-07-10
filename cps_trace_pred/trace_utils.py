"""Trace parsing and formatting helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def safe_json_loads(raw: Any, default: Any) -> Any:
    """Parse JSON robustly, returning ``default`` on failure.

    The processed trace columns occasionally contain doubly-quoted or escaped JSON.
    Strategy:
    1. Try ``json.loads`` directly.
    2. Retry with quote cleanups (doubled quotes and escaped quotes).
    3. If parsing yields a JSON string, attempt one extra decode pass.
    """
    if raw is None or _is_nan(raw):
        return default
    if isinstance(raw, (dict, list)):
        return raw

    text = str(raw).strip()
    if not text:
        return default

    candidates: List[str] = [text]
    if len(text) >= 2 and text[0] == text[-1] == '"':
        candidates.append(text[1:-1])
    candidates.append(text.replace('""', '"'))
    candidates.append(text.replace('\\"', '"'))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except Exception:
                return parsed
        return parsed
    return default


def parse_function_graph_edges(raw: Any) -> List[Tuple[str, str]]:
    """Return a normalized edge list from ``function_graph_edges`` cell content."""
    payload = safe_json_loads(raw, default=[])
    if not isinstance(payload, list):
        return []

    edges: List[Tuple[str, str]] = []
    for pair in payload:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        caller = str(pair[0])
        callee = str(pair[1])
        edges.append((caller, callee))
    return edges


def edge_to_str(caller: str, callee: str) -> str:
    return f"{caller}->{callee}"


def functions_from_edges(edges: Sequence[Tuple[str, str]]) -> List[str]:
    funcs = set()
    for caller, callee in edges:
        funcs.add(str(caller))
        funcs.add(str(callee))
    return sorted(funcs)


def load_layout_map(layout: Optional[Any]) -> Dict[str, str]:
    """Load function-id -> function-name map from layout.json path or dict."""
    if layout is None:
        return {}

    if isinstance(layout, (str, Path)):
        path = Path(layout)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return {}
    elif isinstance(layout, dict):
        payload = layout
    else:
        return {}

    mapping: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and "name" in value:
            mapping[str(key)] = str(value["name"])
    return mapping


def pretty_function(func_id: str, layout_map: Optional[Dict[str, str]] = None) -> str:
    lookup = layout_map or {}
    if func_id in lookup:
        return f"{lookup[func_id]}({func_id})"
    return str(func_id)


def pretty_edge(edge: str, layout_map: Optional[Dict[str, str]] = None) -> str:
    if "->" not in edge:
        return edge
    caller, callee = edge.split("->", 1)
    return f"{pretty_function(caller, layout_map)}->{pretty_function(callee, layout_map)}"

