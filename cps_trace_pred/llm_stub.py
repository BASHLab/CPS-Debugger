"""Deterministic LLM-facing stubs (no API calls)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .trace_utils import load_layout_map, pretty_edge, pretty_function


def summarize_trace_window(win_row: Dict[str, Any], layout_json: Optional[Any] = None) -> str:
    """Create a human-readable summary for one trace-aligned window.

    Expected optional keys in ``win_row``:
    - ``functions_present``: iterable of function IDs
    - ``edges_present``: iterable of ``caller->callee`` strings
    - ``run`` / ``win`` for identifiers
    """
    layout_map = load_layout_map(layout_json)
    run_name = win_row.get("run", "unknown_run")
    win_id = win_row.get("win", "unknown_win")
    functions = list(win_row.get("functions_present", []))
    edges = list(win_row.get("edges_present", []))

    lines = [f"Window summary for run={run_name}, win={win_id}:"]
    if functions:
        pretty_funcs = [pretty_function(str(fid), layout_map) for fid in functions[:10]]
        lines.append(f"- Functions present ({len(functions)}): " + ", ".join(pretty_funcs))
    else:
        lines.append("- Functions present: none")

    if edges:
        pretty_edges = [pretty_edge(str(edge), layout_map) for edge in edges[:10]]
        lines.append(f"- Edges present ({len(edges)}): " + ", ".join(pretty_edges))
    else:
        lines.append("- Edges present: none")
    return "\n".join(lines)


def make_llm_prompt(context: Dict[str, Any]) -> str:
    """Prepare an analysis prompt string for comparing two trace windows."""
    task = context.get("task", "Explain differences between trace windows.")
    summary_a = context.get("window_a_summary", "N/A")
    summary_b = context.get("window_b_summary", "N/A")
    extra = context.get("extra_notes", "")

    prompt = (
        "You are analyzing two CPS trace windows.\n"
        f"Task: {task}\n\n"
        "Window A:\n"
        f"{summary_a}\n\n"
        "Window B:\n"
        f"{summary_b}\n\n"
        "Please explain key differences in function/edge activity and possible controller-state implications."
    )
    if extra:
        prompt += f"\n\nAdditional context:\n{extra}"
    return prompt

