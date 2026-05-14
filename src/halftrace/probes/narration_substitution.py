"""narration_substitution probe: agent describes a tool call instead of making it.

The failure mode is the agent emitting text that names or describes using
a tool, while emitting no tool_call in the same assistant turn. We treat
any assistant turn with at least one tool_call as legitimate — multi-tool
turns and "narrate-then-call" turns both count as action-taking and are
not flagged. The substitution signature we score for is an *empty-of-action*
assistant turn whose text nonetheless invokes tool-name language.

Tool mention patterns are auto-derived from the names of every tool ever
called in the trajectory: the literal name and an underscore-to-space
variant (so `submit_summary` matches both `submit_summary` and
`submit summary`). Tasks that use verb-form tool names (e.g. `lookup` →
`look up`) may want to extend matching by future work; the current
heuristic produces a conservative lower bound on substitution rate.
"""

from __future__ import annotations

from typing import Any

from halftrace.probes.base import Score
from halftrace.trajectory import Trajectory


def narration_substitution(trajectory: Trajectory) -> Score:
    """Score the fraction of assistant text turns that are NOT substitutions.

    A turn is flagged as a substitution when its text mentions any tool
    name from the trajectory and the turn itself has no tool_calls.
    Returns `value=None` when the trajectory has no assistant text turns
    or no tool calls (no pattern set to match against).
    """
    tool_names = {tc.name for _, tc in trajectory.tool_calls()}
    if not tool_names:
        return Score(probe="narration_substitution", value=None, n_observations=0)

    patterns: set[str] = set()
    for name in tool_names:
        base = name.lower()
        patterns.add(base)
        patterns.add(base.replace("_", " "))

    n_text_turns = 0
    n_substituting = 0
    per_turn: list[dict[str, Any]] = []

    for turn in trajectory.turns:
        if turn.role != "assistant":
            continue
        if not turn.content:
            continue
        n_text_turns += 1

        content_lower = turn.content.lower()
        has_tool_call = bool(turn.tool_calls)
        mentioned = sorted(p for p in patterns if p in content_lower)
        is_substitution = bool(mentioned) and not has_tool_call

        if is_substitution:
            n_substituting += 1
        per_turn.append(
            {
                "turn_index": turn.index,
                "mentioned": mentioned,
                "has_tool_call": has_tool_call,
                "is_substitution": is_substitution,
            }
        )

    if n_text_turns == 0:
        return Score(probe="narration_substitution", value=None, n_observations=0)

    return Score(
        probe="narration_substitution",
        value=(n_text_turns - n_substituting) / n_text_turns,
        n_observations=n_text_turns,
        details={"per_turn": per_turn, "n_substituting": n_substituting},
    )
