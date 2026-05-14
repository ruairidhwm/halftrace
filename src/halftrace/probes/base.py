"""Probe protocol and the Score result type.

A Probe is any callable that consumes a Trajectory and returns a Score.
Probes are stateless functions: all configuration is passed at call time
so the same probe can be applied uniformly across a batch of trajectories.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from halftrace.trajectory import Trajectory


class Score(BaseModel):
    """The result of running a probe on a single trajectory.

    `value` is in [0, 1] when the probe made at least one observation, and
    `None` when the trajectory contained nothing to score. Downstream
    fitting code treats `None` as missing data rather than zero.
    """

    model_config = ConfigDict(extra="forbid")

    probe: str
    value: float | None
    n_observations: int = 0
    details: dict[str, Any] = Field(default_factory=lambda: {})


class Probe(Protocol):
    """A callable that scores a trajectory along one capability dimension."""

    def __call__(self, trajectory: Trajectory) -> Score: ...
