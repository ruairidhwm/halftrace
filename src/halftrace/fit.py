"""Halftrace fitting via linear interpolation with bootstrap CIs.

A halftrace is the trajectory length at which a probe's mean score crosses
a threshold (default 0.5). We fit it by linear interpolation between the
two adjacent N values that bracket the threshold, and report a 95% confidence
interval by resampling within-N repetitions with replacement.

Inputs are deliberately plain floats (not Score objects) so the fitting
layer stays general — callers are responsible for filtering out None
values before passing observations in.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict


class Halftrace(BaseModel):
    """Point estimate and bootstrap CI for a probe's halftrace.

    `value` is the interpolated N at which the mean curve crosses
    `threshold`, or None when no such crossing exists in the tested range.
    `ci_low` and `ci_high` are 2.5th/97.5th percentiles of the bootstrap
    distribution, both None when fewer than two bootstrap iterations
    resolved to a finite halftrace.
    """

    model_config = ConfigDict(extra="forbid")

    value: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    threshold: float = 0.5
    n_bootstrap: int = 0
    n_bootstrap_resolved: int = 0


def fit_halftrace(
    observations: Mapping[int, Sequence[float]],
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    rng: random.Random | None = None,
) -> Halftrace:
    """Fit a halftrace from per-N probe scores.

    `observations` maps N (trajectory length) to a sequence of probe scores
    measured at that N. The point estimate is the first N at which the mean
    score crosses `threshold` downward; the CI is the 2.5th-97.5th
    percentile of within-N bootstrap resampling.

    Raises ValueError if `observations` is empty or any N has no scores.
    """
    if not observations:
        raise ValueError("fit_halftrace requires at least one N")
    if not math.isfinite(threshold):
        raise ValueError(f"fit_halftrace: threshold must be finite, got {threshold!r}")
    if n_bootstrap < 0:
        raise ValueError(f"fit_halftrace: n_bootstrap must be non-negative, got {n_bootstrap}")
    for n_value, reps in observations.items():
        if not reps:
            raise ValueError(f"fit_halftrace: N={n_value} has no observations")
        for v in reps:
            if not math.isfinite(v):
                raise ValueError(
                    f"fit_halftrace: N={n_value} has non-finite score {v!r}; "
                    "filter NaN/inf before passing in"
                )

    n_values = sorted(observations)
    materialised: dict[int, list[float]] = {n: list(observations[n]) for n in n_values}
    means = [mean(materialised[n]) for n in n_values]
    point = _interpolate_crossing(n_values, means, threshold)

    if rng is None:
        rng = random.Random()

    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        resampled_means = [
            mean(rng.choices(materialised[n], k=len(materialised[n]))) for n in n_values
        ]
        h = _interpolate_crossing(n_values, resampled_means, threshold)
        if h is not None:
            bootstrap_values.append(h)

    ci_low: float | None = None
    ci_high: float | None = None
    if len(bootstrap_values) >= 2:
        bootstrap_values.sort()
        ci_low = _percentile(bootstrap_values, 2.5)
        ci_high = _percentile(bootstrap_values, 97.5)

    return Halftrace(
        value=point,
        ci_low=ci_low,
        ci_high=ci_high,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        n_bootstrap_resolved=len(bootstrap_values),
    )


def _interpolate_crossing(
    n_values: Sequence[int],
    means: Sequence[float],
    threshold: float,
) -> float | None:
    """First N at which the mean curve crosses strictly below `threshold`.

    Requires `n_values` sorted ascending. Returns None when no adjacent
    pair brackets the threshold.
    """
    for i in range(len(n_values) - 1):
        y0, y1 = means[i], means[i + 1]
        if y0 >= threshold > y1:
            x0, x1 = n_values[i], n_values[i + 1]
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return None


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted sequence."""
    if not sorted_values:
        raise ValueError("_percentile: empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
