"""Tests for halftrace fitting and bootstrap CIs."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from halftrace import Halftrace, fit_halftrace

_FLOAT_TOL = 1e-9


def _seeded_rng() -> random.Random:
    return random.Random(0xC0FFEE)


class TestPointEstimate:
    """The interpolated crossing on the mean curve."""

    def test_simple_two_point_crossing_interpolates_linearly(self) -> None:
        # At N=10 mean is 1.0, at N=20 mean is 0.0 → crossing at N=15.
        h = fit_halftrace({10: [1.0], 20: [0.0]}, n_bootstrap=0)
        assert h.value is not None
        assert abs(h.value - 15.0) < _FLOAT_TOL

    def test_crossing_at_exact_lower_point_returns_that_point(self) -> None:
        # y0 == threshold → interpolation lands on x0.
        h = fit_halftrace({10: [0.5], 20: [0.0]}, n_bootstrap=0)
        assert h.value == 10.0

    def test_realistic_decay_curve_interpolates_between_bracketing_points(self) -> None:
        # mean curve: 1.0, 0.9, 0.6, 0.3 at N = 5, 10, 50, 100
        # crossing between N=50 (0.6) and N=100 (0.3): 50 + (0.5-0.6)*(100-50)/(0.3-0.6)
        #                                            = 50 + (-0.1)*(50)/(-0.3) = 50 + 16.666...
        h = fit_halftrace(
            {5: [1.0], 10: [0.9], 50: [0.6], 100: [0.3]},
            n_bootstrap=0,
        )
        assert h.value is not None
        assert abs(h.value - (50 + 50 / 3)) < _FLOAT_TOL

    def test_first_crossing_wins_for_non_monotonic_curves(self) -> None:
        # Dips below threshold then recovers; the first crossing is what we report.
        h = fit_halftrace({5: [0.9], 10: [0.3], 20: [0.8], 30: [0.2]}, n_bootstrap=0)
        assert h.value is not None
        # crossing between N=5 (0.9) and N=10 (0.3)
        # 5 + (0.5-0.9)*(10-5)/(0.3-0.9) = 5 + (-0.4)*(5)/(-0.6) = 5 + 10/3
        assert abs(h.value - (5 + 10 / 3)) < _FLOAT_TOL

    def test_custom_threshold_is_respected(self) -> None:
        # Crossing 0.75 between N=10 (1.0) and N=20 (0.5):
        # 10 + (0.75-1.0)*(20-10)/(0.5-1.0) = 10 + (-0.25)*(10)/(-0.5) = 15
        h = fit_halftrace({10: [1.0], 20: [0.5]}, threshold=0.75, n_bootstrap=0)
        assert h.value is not None
        assert abs(h.value - 15.0) < _FLOAT_TOL
        assert h.threshold == 0.75


class TestNoCrossing:
    """Curves that never cross the threshold report value=None."""

    def test_all_above_threshold_returns_none(self) -> None:
        h = fit_halftrace({5: [0.9], 10: [0.8], 20: [0.7]}, n_bootstrap=0)
        assert h.value is None

    def test_all_below_threshold_returns_none(self) -> None:
        h = fit_halftrace({5: [0.3], 10: [0.2], 20: [0.1]}, n_bootstrap=0)
        assert h.value is None

    def test_single_n_value_cannot_interpolate(self) -> None:
        h = fit_halftrace({10: [1.0, 0.0]}, n_bootstrap=0)
        assert h.value is None


class TestBootstrap:
    """Bootstrap resampling produces a CI bracketing the point estimate."""

    def test_seeded_rng_is_reproducible(self) -> None:
        obs = {5: [1.0, 1.0, 0.9], 10: [0.8, 0.7, 0.6], 20: [0.3, 0.2, 0.1]}
        h1 = fit_halftrace(obs, rng=_seeded_rng(), n_bootstrap=200)
        h2 = fit_halftrace(obs, rng=_seeded_rng(), n_bootstrap=200)
        assert h1 == h2

    def test_ci_brackets_the_point_estimate_for_clean_signal(self) -> None:
        obs = {
            5: [1.0, 1.0, 1.0, 0.9, 1.0],
            10: [0.9, 1.0, 0.9, 0.8, 0.9],
            20: [0.6, 0.5, 0.7, 0.6, 0.5],
            40: [0.2, 0.3, 0.1, 0.2, 0.2],
        }
        h = fit_halftrace(obs, rng=_seeded_rng(), n_bootstrap=500)
        assert h.value is not None
        assert h.ci_low is not None and h.ci_high is not None
        assert h.ci_low <= h.value <= h.ci_high

    def test_ci_is_none_when_no_bootstrap_runs(self) -> None:
        h = fit_halftrace({10: [1.0], 20: [0.0]}, n_bootstrap=0)
        assert h.ci_low is None
        assert h.ci_high is None
        assert h.n_bootstrap == 0
        assert h.n_bootstrap_resolved == 0

    def test_single_rep_per_n_collapses_ci_to_point(self) -> None:
        # With k=1 reps, resampling with replacement always returns the same value,
        # so every bootstrap iteration produces the same halftrace.
        h = fit_halftrace({10: [1.0], 20: [0.0]}, rng=_seeded_rng(), n_bootstrap=100)
        assert h.value is not None
        assert h.ci_low is not None and h.ci_high is not None
        assert abs(h.ci_low - h.value) < _FLOAT_TOL
        assert abs(h.ci_high - h.value) < _FLOAT_TOL

    def test_n_bootstrap_resolved_counts_only_finite_halftraces(self) -> None:
        # Borderline case: at N=5 mean is exactly 0.5, at N=10 mean is 0.4. Some
        # bootstrap resamples will land at all-1.0 (no crossing). Resolved < total.
        obs = {5: [1.0, 0.0], 10: [0.4, 0.4]}
        h = fit_halftrace(obs, rng=_seeded_rng(), n_bootstrap=500)
        assert h.n_bootstrap == 500
        assert 0 < h.n_bootstrap_resolved < 500


class TestValidation:
    """Bad input raises rather than silently producing nonsense."""

    def test_empty_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one N"):
            fit_halftrace({})

    def test_empty_reps_at_one_n_raises(self) -> None:
        with pytest.raises(ValueError, match="no observations"):
            fit_halftrace({10: [1.0], 20: []})

    def test_nan_score_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite score"):
            fit_halftrace({10: [1.0, float("nan")], 20: [0.0]})

    def test_inf_score_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite score"):
            fit_halftrace({10: [1.0], 20: [float("inf")]})

    def test_non_finite_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be finite"):
            fit_halftrace({10: [1.0], 20: [0.0]}, threshold=float("nan"))

    def test_negative_n_bootstrap_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap must be non-negative"):
            fit_halftrace({10: [1.0], 20: [0.0]}, n_bootstrap=-1)


class TestHalftraceSchema:
    """The Halftrace model is a strict pydantic schema."""

    def test_halftrace_round_trips_through_json(self) -> None:
        h = fit_halftrace({10: [1.0], 20: [0.0]}, rng=_seeded_rng(), n_bootstrap=50)
        restored = Halftrace.model_validate_json(h.model_dump_json())
        assert restored == h

    def test_halftrace_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Halftrace.model_validate({"value": 1.0, "mystery": True})

    def test_halftrace_value_can_be_none(self) -> None:
        h = Halftrace(value=None)
        assert h.value is None
        assert h.threshold == 0.5
