"""tool_repetition probe: avoidance of re-calling tools with the same arguments.

Unlike state_amnesia and instruction_decay, this probe needs no task
annotations: it inspects the trajectory's tool_calls directly. Tasks
still influence the result by their tool design — tasks that lend
themselves to natural repetition will produce more duplicate calls.

The score is the fraction of tool calls that are first-occurrences of
their `(name, args)` key. Higher is better; the floor for any non-empty
trajectory is `1/n_calls` (the first call is never a duplicate of itself).
"""

from __future__ import annotations

import json
from typing import Any

from halftrace.probes.base import Score
from halftrace.trajectory import Trajectory


def tool_repetition(trajectory: Trajectory) -> Score:
    """Score the fraction of tool calls that are unique by (name, args).

    Args are normalised with sorted JSON keys so dicts that differ only
    in key order are still detected as duplicates. Returns `value=None`
    when the trajectory contains no tool calls.
    """
    seen: set[tuple[str, str]] = set()
    total = 0
    n_duplicates = 0
    per_call: list[dict[str, Any]] = []

    for turn_index, tc in trajectory.tool_calls():
        total += 1
        key = (tc.name, json.dumps(tc.args, sort_keys=True))
        is_duplicate = key in seen
        if is_duplicate:
            n_duplicates += 1
        else:
            seen.add(key)
        per_call.append(
            {
                "turn_index": turn_index,
                "name": tc.name,
                "is_duplicate": is_duplicate,
            }
        )

    if total == 0:
        return Score(probe="tool_repetition", value=None, n_observations=0)

    return Score(
        probe="tool_repetition",
        value=(total - n_duplicates) / total,
        n_observations=total,
        details={"per_call": per_call, "n_duplicates": n_duplicates},
    )
