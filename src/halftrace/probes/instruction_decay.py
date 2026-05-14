"""instruction_decay probe: adherence to a system-prompt rule over time.

Tasks declare a single rule in `Trajectory.metadata["instruction_decay"]`:

    {"rule_id": "end_with_marker", "params": {"marker": "[[END]]"}}

The probe iterates assistant turns that have text content and reports the
fraction that satisfy the rule. The default rule `end_with_marker` checks
that the response (after a trailing-whitespace trim) ends with the
configured marker. Callers can pass a custom `matcher` to score any other
rule shape against the same params dict.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from halftrace.probes.base import Score
from halftrace.trajectory import Trajectory

Matcher = Callable[[str, dict[str, Any]], bool]


def _ends_with_marker(content: str, params: dict[str, Any]) -> bool:
    marker = params.get("marker")
    if not isinstance(marker, str):
        raise ValueError(
            f"end_with_marker rule requires a 'marker' string param, got {marker!r}"
        )
    return content.rstrip().endswith(marker)


def _starts_with_pattern(content: str, params: dict[str, Any]) -> bool:
    pattern = params.get("pattern")
    if not isinstance(pattern, str):
        raise ValueError(
            f"starts_with_pattern rule requires a 'pattern' string param, got {pattern!r}"
        )
    return re.match(pattern, content.lstrip()) is not None


_BUILTIN_MATCHERS: dict[str, Matcher] = {
    "end_with_marker": _ends_with_marker,
    "starts_with_pattern": _starts_with_pattern,
}


def instruction_decay(
    trajectory: Trajectory,
    *,
    matcher: Matcher | None = None,
) -> Score:
    """Score adherence to the trajectory's instruction_decay rule.

    Returns a Score whose `value` is the fraction of assistant text turns
    that satisfy the rule. Returns `value=None` when no rule is annotated
    or no assistant text turns are present.

    Raises ValueError if the annotation is malformed or names an unknown
    builtin rule_id (when no custom matcher is supplied).
    """
    raw_rule = trajectory.metadata.get("instruction_decay")
    if raw_rule is None:
        return Score(probe="instruction_decay", value=None, n_observations=0)
    if not isinstance(raw_rule, dict):
        raise ValueError(
            f"instruction_decay metadata must be a dict, got {type(raw_rule).__name__}"
        )
    rule = cast("dict[str, Any]", raw_rule)

    rule_id = rule.get("rule_id")
    raw_params = rule.get("params", {})
    if not isinstance(raw_params, dict):
        raise ValueError(
            f"instruction_decay 'params' must be a dict, got {type(raw_params).__name__}"
        )
    params = cast("dict[str, Any]", raw_params)

    if matcher is None:
        if not isinstance(rule_id, str):
            raise ValueError(
                f"instruction_decay 'rule_id' must be a string, got {rule_id!r}"
            )
        if rule_id not in _BUILTIN_MATCHERS:
            raise ValueError(
                f"instruction_decay: unknown rule_id {rule_id!r}; "
                f"known: {sorted(_BUILTIN_MATCHERS)}"
            )
        matcher = _BUILTIN_MATCHERS[rule_id]

    follows = 0
    per_turn: list[dict[str, Any]] = []
    n_obs = 0
    for turn in trajectory.turns:
        if turn.role != "assistant":
            continue
        if not turn.content:
            continue
        ok = matcher(turn.content, params)
        per_turn.append({"turn_index": turn.index, "follows": ok})
        if ok:
            follows += 1
        n_obs += 1

    if n_obs == 0:
        return Score(probe="instruction_decay", value=None, n_observations=0)

    return Score(
        probe="instruction_decay",
        value=follows / n_obs,
        n_observations=n_obs,
        details={"per_turn": per_turn},
    )
