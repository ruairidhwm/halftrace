"""state_amnesia probe: retention of facts introduced earlier in the trajectory.

We create a 'plant' which is a piece of information that the model should remember
into the trajectory. For example: 'user turn says the door is red'. The recall then
asks for the fact back 'what colour is the door?'. We then check whether the planted
fact appears.

Tasks annotate plant and recall turns via `Turn.metadata["state_amnesia"]`:

    Plant:  {"role": "plant",  "fact_id": "<id>", "fact": "<text>"}
    Recall: {"role": "recall", "fact_ids": ["<id>", "<id>", ...]}

A recall annotation may reference one or more planted fact_ids. The probe
pairs each (recall_turn, fact_id) with the next assistant text turn after
the recall and reports the fraction of recalls answered correctly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from halftrace.probes.base import Score
from halftrace.trajectory import Trajectory


def _default_matcher(planted: str, response: str) -> bool:
    """Case-insensitive substring containment after whitespace normalisation."""
    norm_planted = " ".join(planted.lower().split())
    norm_response = " ".join(response.lower().split())
    return norm_planted in norm_response


def state_amnesia(
    trajectory: Trajectory,
    *,
    matcher: Callable[[str, str], bool] = _default_matcher,
) -> Score:
    """Score retention of facts planted earlier in the trajectory.

    Returns a Score in [0, 1] equal to the fraction of recall queries the
    agent answered correctly, or `value=None` when no recall annotations
    are present. Recalls whose subsequent assistant response is missing
    are counted as incorrect with `reason="no_response"` in details.

    Raises ValueError if the trajectory contains a malformed annotation
    or a recall references a fact_id that was never planted.
    """
    plants: dict[str, str] = {}
    recalls: list[tuple[int, str]] = []

    for turn in trajectory.turns:
        raw = turn.metadata.get("state_amnesia")
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(
                f"state_amnesia annotation at turn {turn.index} must be a dict, "
                f"got {type(raw).__name__}"
            )
        annotation = cast("dict[str, Any]", raw)
        role = annotation.get("role")
        if role == "plant":
            fact_id = annotation.get("fact_id")
            if not isinstance(fact_id, str):
                raise ValueError(
                    f"state_amnesia plant at turn {turn.index} is missing a string fact_id"
                )
            fact = annotation.get("fact")
            if not isinstance(fact, str):
                raise ValueError(
                    f"state_amnesia plant at turn {turn.index} is missing a string fact"
                )
            plants[fact_id] = fact
        elif role == "recall":
            raw_fact_ids = annotation.get("fact_ids")
            if not isinstance(raw_fact_ids, list):
                raise ValueError(
                    f"state_amnesia recall at turn {turn.index} is missing a "
                    "list 'fact_ids'"
                )
            for fid in cast("list[Any]", raw_fact_ids):
                if not isinstance(fid, str):
                    raise ValueError(
                        f"state_amnesia recall at turn {turn.index} has a "
                        f"non-string fact_id: {fid!r}"
                    )
                recalls.append((turn.index, fid))
        else:
            raise ValueError(
                f"state_amnesia annotation at turn {turn.index} has unknown role {role!r}"
            )

    if not recalls:
        return Score(probe="state_amnesia", value=None, n_observations=0)

    correct = 0
    per_fact: list[dict[str, Any]] = []

    for recall_index, fact_id in recalls:
        if fact_id not in plants:
            raise ValueError(
                f"state_amnesia recall at turn {recall_index} references "
                f"unplanted fact_id {fact_id!r}"
            )
        response = _next_assistant_content(trajectory, recall_index)
        if response is None:
            per_fact.append({"fact_id": fact_id, "correct": False, "reason": "no_response"})
            continue
        is_correct = matcher(plants[fact_id], response)
        per_fact.append({"fact_id": fact_id, "correct": is_correct})
        if is_correct:
            correct += 1

    return Score(
        probe="state_amnesia",
        value=correct / len(recalls),
        n_observations=len(recalls),
        details={"per_fact": per_fact},
    )


def _next_assistant_content(trajectory: Trajectory, after_index: int) -> str | None:
    for turn in trajectory.turns:
        if turn.index <= after_index:
            continue
        if turn.role == "assistant" and turn.content:
            return turn.content
    return None
