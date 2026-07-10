"""CPS Agent-Computer Interface: tools for the debugging agent.

Provides structured access to source code, sensor data, execution traces,
and anomaly reports. Inspired by SWE-agent's ACI design.

The tools are defined both as Python methods (for direct use) and as
Anthropic tool_use JSON schemas (for the messages API).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Function ID -> Name mapping (from layout.json) ──────────────────────

FUNC_NAMES = {
    "22": "tick", "33": "pou_main", "34": "turn_on_with_delay",
    "35": "turn_off_with_delay", "37": "pou_general_machine",
    "38": "pou_general_drive", "39": "pou_standup_rel",
    "40": "pou_lqr_sim", "42": "pou_drive_control_word",
    "63": "fmin", "64": "fmax", "102": "printf_core",
    "108": "memcpy", "109": "memset",
}


# ── Anthropic tool_use schemas ──────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "read_source",
        "description": (
            "View the DEPLOYED controller source code (currently running). "
            "Use function_name to search for a function definition across all "
            ".c files, or file_name to view a specific file (optionally with "
            "line_start/line_end to limit output). This is the code that may "
            "contain a bug."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Search for this function definition across all source files.",
                },
                "file_name": {
                    "type": "string",
                    "description": "View this specific file (e.g. 'lqrsim.c' or 'pendulum_controller/lqrsim.c').",
                },
                "line_start": {
                    "type": "integer",
                    "description": "First line to show (1-indexed). Requires file_name.",
                },
                "line_end": {
                    "type": "integer",
                    "description": "Last line to show (1-indexed). Requires file_name.",
                },
            },
        },
    },
    {
        "name": "read_sensor",
        "description": (
            "Query datalayer sensor readings. With no arguments, lists available "
            "channels. With channel, shows summary statistics. With channel + "
            "window_start/window_end, shows values for that time range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Sensor channel name (e.g. 'dl_angle_mean').",
                },
                "window_start": {"type": "integer", "description": "Start window index."},
                "window_end": {"type": "integer", "description": "End window index."},
            },
        },
    },
    {
        "name": "read_trace",
        "description": (
            "Query Wasm execution trace edge counts. Edges are named "
            "func_id:from_pc:to_pc. With no arguments, lists all edges grouped "
            "by function. With edge_id, shows statistics. Optional window range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edge_id": {
                    "type": "string",
                    "description": "Edge identifier (e.g. '40:44:85').",
                },
                "window_start": {"type": "integer"},
                "window_end": {"type": "integer"},
            },
        },
    },
    {
        "name": "read_reference",
        "description": (
            "View the REFERENCE source code from the git repository (known-good "
            "version). Compare against read_source (deployed version) to find "
            "modifications. Same arguments as read_source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Search for this function definition.",
                },
                "file_name": {
                    "type": "string",
                    "description": "View this specific file.",
                },
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_anomaly_report",
        "description": (
            "Get the Stage 1 anomaly diagnostic report: which residual channels "
            "spiked, which function/edges are affected, severity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "submit_diagnosis",
        "description": (
            "Submit your final diagnosis and code fix. Call this exactly once "
            "when you are confident in your answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fault_type": {
                    "type": "string",
                    "enum": ["parameter_fault", "logic_fault", "timing_fault", "sensor_fault"],
                    "description": "Category of the fault.",
                },
                "affected_function": {
                    "type": "string",
                    "description": "Name of the function containing the bug.",
                },
                "root_cause": {
                    "type": "string",
                    "description": "One-sentence description of the root cause.",
                },
                "fix_file": {
                    "type": "string",
                    "description": "File containing the bug (e.g. 'lqrsim.c').",
                },
                "fix_old": {
                    "type": "string",
                    "description": "Exact buggy text to replace.",
                },
                "fix_new": {
                    "type": "string",
                    "description": "Corrected replacement text.",
                },
            },
            "required": ["fault_type", "affected_function", "root_cause",
                         "fix_file", "fix_old", "fix_new"],
        },
    },
]


class CPS_ACI:
    """Agent-Computer Interface for CPS debugging."""

    def __init__(
        self,
        source_dir: Path,
        aligned_data: pd.DataFrame,
        physical_cols: List[str],
        branch_cols: List[str],
        anomaly_observation: Optional[Dict] = None,
        reference_dir: Optional[Path] = None,
    ):
        self.source_dir = Path(source_dir)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.data = aligned_data
        self.physical_cols = physical_cols
        self.branch_cols = branch_cols
        self.observation = anomaly_observation

    # ── Tool implementations ────────────────────────────────────────

    def read_source(
        self,
        function_name: str = None,
        file_name: str = None,
        line_start: int = None,
        line_end: int = None,
    ) -> str:
        if function_name and not file_name:
            # Search all .c files for the function definition
            for f in sorted(self.source_dir.rglob("*.c")):
                lines = f.read_text().split("\n")
                for i, line in enumerate(lines):
                    # Match function definition (name followed by '(')
                    if function_name in line and "(" in line and not line.strip().startswith("//"):
                        # Check it looks like a definition, not a call
                        stripped = line.lstrip()
                        is_def = (
                            stripped.startswith("void ")
                            or stripped.startswith("double ")
                            or stripped.startswith("bool ")
                            or stripped.startswith("int ")
                            or stripped.startswith("static ")
                            or stripped.startswith("uint")
                            or stripped.startswith("const ")
                        )
                        if not is_def:
                            continue
                        # Show function with context
                        start = max(0, i - 3)
                        end = min(len(lines), i + 60)
                        rel = f.relative_to(self.source_dir)
                        result = f"=== {rel}, lines {start+1}-{end} ===\n"
                        for j in range(start, end):
                            result += f"{j+1:4d} | {lines[j]}\n"
                        return result
            return f"Function '{function_name}' not found in source files."

        elif file_name:
            # Try exact path first, then search
            candidates = list(self.source_dir.rglob(file_name))
            if not candidates:
                available = [str(f.relative_to(self.source_dir)) for f in self.source_dir.rglob("*.c")]
                return f"File '{file_name}' not found. Available:\n" + "\n".join(f"  {a}" for a in sorted(available))
            f = candidates[0]
            lines = f.read_text().split("\n")
            start = (line_start - 1) if line_start else 0
            end = line_end if line_end else len(lines)
            start = max(0, start)
            end = min(len(lines), end)
            rel = f.relative_to(self.source_dir)
            result = f"=== {rel}, lines {start+1}-{end} of {len(lines)} ===\n"
            for j in range(start, end):
                result += f"{j+1:4d} | {lines[j]}\n"
            return result

        else:
            # List all source files
            result = "Available source files:\n"
            for f in sorted(self.source_dir.rglob("*.c")):
                rel = f.relative_to(self.source_dir)
                n = sum(1 for _ in open(f))
                result += f"  {rel} ({n} lines)\n"
            for f in sorted(self.source_dir.rglob("*.h")):
                rel = f.relative_to(self.source_dir)
                n = sum(1 for _ in open(f))
                result += f"  {rel} ({n} lines)\n"
            return result

    def read_sensor(
        self,
        channel: str = None,
        window_start: int = None,
        window_end: int = None,
    ) -> str:
        if channel is None:
            return "Available sensor channels:\n" + "\n".join(
                f"  {c}" for c in self.physical_cols
            )

        if channel not in self.data.columns:
            close = [c for c in self.physical_cols if channel in c]
            return (
                f"Channel '{channel}' not found."
                + (f" Did you mean: {close}" if close else "")
            )

        if window_start is not None and window_end is not None:
            seg = self.data[channel].iloc[window_start:window_end]
            vals = seg.values.tolist()
            preview = vals[:30]
            return (
                f"{channel} (windows {window_start}-{window_end}):\n"
                f"  mean={seg.mean():.6f}, std={seg.std():.6f}\n"
                f"  min={seg.min():.6f}, max={seg.max():.6f}\n"
                f"  values (first 30): {preview}"
            )

        col = self.data[channel]
        return (
            f"{channel} (full dataset, {len(col)} windows):\n"
            f"  mean={col.mean():.6f}, std={col.std():.6f}\n"
            f"  min={col.min():.6f}, max={col.max():.6f}"
        )

    def read_trace(
        self,
        edge_id: str = None,
        window_start: int = None,
        window_end: int = None,
    ) -> str:
        if edge_id is None:
            by_func = defaultdict(list)
            for col in self.branch_cols:
                fid = col.split(":")[0]
                by_func[fid].append(col)
            result = "Execution trace edges (func_id:from_pc:to_pc):\n"
            for fid in sorted(by_func.keys(), key=int):
                fname = FUNC_NAMES.get(fid, f"func_{fid}")
                edges = by_func[fid]
                result += f"\n  Function {fid} ({fname}, {len(edges)} edges):\n"
                for e in sorted(edges, key=lambda x: int(x.split(":")[1])):
                    mean = self.data[e].mean()
                    result += f"    {e}: mean_count={mean:.2f}\n"
            return result

        matching = [c for c in self.branch_cols if edge_id in c]
        if not matching:
            return f"Edge '{edge_id}' not found in trace data."

        col = matching[0]
        if window_start is not None and window_end is not None:
            seg = self.data[col].iloc[window_start:window_end]
            return (
                f"Edge {col} (windows {window_start}-{window_end}):\n"
                f"  mean={seg.mean():.2f}, std={seg.std():.2f}\n"
                f"  min={seg.min():.2f}, max={seg.max():.2f}\n"
                f"  values (first 30): {seg.values.tolist()[:30]}"
            )
        series = self.data[col]
        return (
            f"Edge {col} (full dataset, {len(series)} windows):\n"
            f"  mean={series.mean():.2f}, std={series.std():.2f}\n"
            f"  min={series.min():.2f}, max={series.max():.2f}"
        )

    def get_anomaly_report(self) -> str:
        if self.observation is None:
            return "No anomaly observation loaded."
        alert = self.observation["alert"]
        result = "=== ANOMALY REPORT ===\n\n"
        result += f"Status: anomaly detected\n"
        result += f"Trace residual: {alert['trace_residual']}\n"
        result += f"Sensor residual: {alert['sensor_residual']}\n"
        result += f"Dynamics residual: {alert['dynamics_residual']}\n"
        result += f"Affected function (trace-based): {alert['affected_function']}\n"
        result += f"Affected edges: {alert['affected_edges']}\n"
        result += f"Time range: {alert['timestamp_range']}\n"

        if self.observation.get("edge_summary"):
            result += "\nEdge deviations from normal:\n"
            for edge, stats in self.observation["edge_summary"].items():
                result += (
                    f"  {edge}: normal_mean={stats['normal_mean']:.2f}, "
                    f"observed={stats['observed_mean']:.2f}, "
                    f"deviation={stats['deviation_sigma']:.1f}sigma\n"
                )
        return result

    # ── Dispatch tool calls from Anthropic API ──────────────────────

    def read_reference(self, **kwargs) -> str:
        """Read from the reference (known-good) source directory."""
        if self.reference_dir is None:
            return "Reference source not available."
        # Temporarily swap source_dir to read from reference
        orig = self.source_dir
        self.source_dir = self.reference_dir
        try:
            return self.read_source(**kwargs)
        finally:
            self.source_dir = orig

    def dispatch(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool call and return the text result."""
        if tool_name == "read_source":
            return self.read_source(**tool_input)
        elif tool_name == "read_reference":
            return self.read_reference(**tool_input)
        elif tool_name == "read_sensor":
            return self.read_sensor(**tool_input)
        elif tool_name == "read_trace":
            return self.read_trace(**tool_input)
        elif tool_name == "get_anomaly_report":
            return self.get_anomaly_report()
        elif tool_name == "submit_diagnosis":
            return json.dumps(tool_input, indent=2)
        else:
            return f"Unknown tool: {tool_name}"
