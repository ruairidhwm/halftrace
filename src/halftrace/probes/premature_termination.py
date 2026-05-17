"""premature_termination probe: did the agent finish before completing required work?

The agent shouldn't *decide* a long task is complete before the work is
actually done. This probe scores how much of the required work was
completed before the agent submitted its terminating tool call.

Tasks annotate the trajectory with two pieces of information:

- `Trajectory.metadata["premature_termination"]`: a dict with the keys
  `n_required` (positive int) and `n_completed` (non-negative int). Tasks
  should populate this when the terminating tool is called.

The probe reports `value = min(n_completed / n_required, 1.0)`. Returns
`value=None` when no annotation is present, e.g. on a trajectory that
ran to its natural end without the task recording termination data.

This probe is mechanically simple. The hard part is task design — the
probe will only produce non-trivial scores on tasks where the agent has
a real choice about when to terminate. Tasks that enumerate the work
upfront (e.g. `find_and_synthesise`) foreclose this failure mode.
"""

from __future__ import annotations

from typing import Any, cast

from halftrace.probes.base import Score
from halftrace.trajectory import Trajectory


def premature_termination(trajectory: Trajectory) -> Score:
    """Score the agent's completion of required work before termination.

    Returns `value` in [0, 1] where 1.0 means all required work was
    completed before the agent submitted, and 0.0 means it terminated
    without completing any of it. Returns `value=None` when no
    `premature_termination` annotation is present.

    Looks for the annotation at trajectory-level first, then falls back
    to the last turn that carries one (which is how tool-result-driven
    tasks like `find_max` record it: the terminating tool call's
    response is the natural place to put completion data).

    Raises ValueError if the annotation is malformed or `n_required <= 0`.
    """
    annotation = _resolve_annotation(trajectory)
    if annotation is None:
        return Score(probe="premature_termination", value=None, n_observations=0)

    raw_required = annotation.get("n_required")
    raw_completed = annotation.get("n_completed")
    if not isinstance(raw_required, int) or raw_required <= 0:
        raise ValueError(
            f"premature_termination 'n_required' must be a positive int, "
            f"got {raw_required!r}"
        )
    if not isinstance(raw_completed, int) or raw_completed < 0:
        raise ValueError(
            f"premature_termination 'n_completed' must be a non-negative int, "
            f"got {raw_completed!r}"
        )

    value = min(raw_completed / raw_required, 1.0)
    return Score(
        probe="premature_termination",
        value=value,
        n_observations=1,
        details={"n_required": raw_required, "n_completed": raw_completed},
    )


def _resolve_annotation(trajectory: Trajectory) -> dict[str, Any] | None:
    """Find the premature_termination annotation, preferring trajectory-level."""
    raw = trajectory.metadata.get("premature_termination")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(
                f"premature_termination metadata must be a dict, "
                f"got {type(raw).__name__}"
            )
        return cast("dict[str, Any]", raw)
    # Fall back to the most recent turn that carries the annotation.
    for turn in reversed(trajectory.turns):
        turn_raw = turn.metadata.get("premature_termination")
        if turn_raw is None:
            continue
        if not isinstance(turn_raw, dict):
            raise ValueError(
                f"premature_termination metadata at turn {turn.index} must be a "
                f"dict, got {type(turn_raw).__name__}"
            )
        return cast("dict[str, Any]", turn_raw)
    return None
