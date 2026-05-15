"""Compare two sets of compliance profiles.

The diff-mode workflow is the iterative loop most agent developers
actually want: "I changed my system prompt — did the change improve
compliance?". This module compares two probe-keyed dicts of
`ComplianceProfile` and emits a structured per-probe diff that the CLI
can render side-by-side.

Direction is decided by the delta in `commit_probability` (the headline
scalar that exists for every shape). Shape changes are surfaced
separately as a boolean for human inspection.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from halftrace.fit import ComplianceProfile

Direction = Literal[
    "improved",
    "regressed",
    "unchanged",
    "appeared",
    "disappeared",
    "missing",
]


class ProbeComparison(BaseModel):
    """One probe's before/after profile pair and the diff between them."""

    model_config = ConfigDict(extra="forbid")

    probe: str
    before: ComplianceProfile | None
    after: ComplianceProfile | None
    direction: Direction
    commit_probability_delta: float | None
    shape_changed: bool


def compare_profiles(
    before: dict[str, ComplianceProfile | None],
    after: dict[str, ComplianceProfile | None],
    *,
    delta_threshold: float = 0.05,
) -> list[ProbeComparison]:
    """Produce one ProbeComparison per probe across the union of both dicts.

    `delta_threshold` is the change in commit_probability above which the
    direction is `improved` or `regressed` rather than `unchanged`. The
    default (0.05) matches what's typically reproducible at small rep counts.
    """
    probe_names = sorted(set(before) | set(after))
    comparisons: list[ProbeComparison] = []
    for probe_name in probe_names:
        b = before.get(probe_name)
        a = after.get(probe_name)
        comparisons.append(
            _compare_one(probe_name, b, a, delta_threshold=delta_threshold)
        )
    return comparisons


def _compare_one(
    probe: str,
    before: ComplianceProfile | None,
    after: ComplianceProfile | None,
    *,
    delta_threshold: float,
) -> ProbeComparison:
    if before is None and after is None:
        return ProbeComparison(
            probe=probe,
            before=None,
            after=None,
            direction="missing",
            commit_probability_delta=None,
            shape_changed=False,
        )
    if before is None:
        assert after is not None
        return ProbeComparison(
            probe=probe,
            before=None,
            after=after,
            direction="appeared",
            commit_probability_delta=after.commit_probability,
            shape_changed=True,
        )
    if after is None:
        return ProbeComparison(
            probe=probe,
            before=before,
            after=None,
            direction="disappeared",
            commit_probability_delta=-before.commit_probability,
            shape_changed=True,
        )

    delta = after.commit_probability - before.commit_probability
    shape_changed = before.shape != after.shape
    if delta > delta_threshold:
        direction: Direction = "improved"
    elif delta < -delta_threshold:
        direction = "regressed"
    else:
        direction = "unchanged"
    return ProbeComparison(
        probe=probe,
        before=before,
        after=after,
        direction=direction,
        commit_probability_delta=delta,
        shape_changed=shape_changed,
    )
