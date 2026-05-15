"""Tests for analyse_compliance: shape classification + commit probability + halftrace."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from halftrace import ComplianceProfile, analyse_compliance


class TestShapeClassification:
    """The classifier should label each of the six shapes correctly."""

    def test_all_means_above_floor_is_perfect(self) -> None:
        profile = analyse_compliance({5: [1.0, 1.0, 1.0], 10: [1.0, 0.97, 1.0]})
        assert profile.shape == "perfect"
        assert profile.commit_probability >= 0.83  # 5/6 above 0.95

    def test_all_means_below_ceiling_is_abandoned(self) -> None:
        profile = analyse_compliance({5: [0.0, 0.0, 0.03], 10: [0.0, 0.05, 0.0]})
        assert profile.shape == "abandoned"
        assert profile.commit_probability == 0.0

    def test_high_within_n_variance_is_bimodal(self) -> None:
        # Sonnet 4.6 counter-rule data: at N=10, two reps coinflipped to 0.18 and 1.0
        profile = analyse_compliance({5: [1.0, 1.0], 10: [0.18, 1.0]})
        assert profile.shape == "bimodal"
        assert profile.max_within_n_variance > 0.15

    def test_intermediate_means_with_low_variance_is_categorical(self) -> None:
        # The discovery-mode finding: stable ~0.5 with near-zero variance
        profile = analyse_compliance({10: [0.545, 0.545], 25: [0.52, 0.52]})
        assert profile.shape == "categorical"
        assert profile.max_within_n_variance < 0.01

    def test_monotone_decreasing_is_gradient(self) -> None:
        profile = analyse_compliance({5: [1.0], 10: [0.7], 20: [0.3]})
        assert profile.shape == "gradient"

    def test_non_monotone_intermediate_high_variance_is_unclassified(self) -> None:
        # Means bounce around with intermediate values but high variance — none of the others fit
        profile = analyse_compliance({5: [0.8, 0.2], 10: [0.3, 0.9], 20: [0.6, 0.1]})
        # high variance → bimodal first
        assert profile.shape == "bimodal"


class TestCommitProbability:
    """commit_probability is the headline scalar for all shapes."""

    def test_default_threshold_is_strict(self) -> None:
        # 0.9 is below the default commit_threshold of 0.95
        profile = analyse_compliance({5: [0.9, 0.9, 0.9]})
        assert profile.commit_probability == 0.0

    def test_threshold_is_inclusive_lower_bound(self) -> None:
        profile = analyse_compliance({5: [0.95, 0.96]})
        assert profile.commit_probability == 1.0

    def test_custom_threshold(self) -> None:
        profile = analyse_compliance(
            {5: [0.6, 0.8, 0.9]}, commit_threshold=0.7
        )
        # 0.8 and 0.9 are >= 0.7, 0.6 is not → 2/3
        assert abs(profile.commit_probability - 2 / 3) < 1e-9


class TestHalftrace:
    """Halftrace is only populated when shape is `gradient`."""

    def test_gradient_shape_populates_halftrace(self) -> None:
        profile = analyse_compliance({5: [1.0], 10: [0.7], 20: [0.3]})
        assert profile.shape == "gradient"
        assert profile.halftrace is not None
        # Crossing of 0.5 between N=10 (0.7) and N=20 (0.3): 10 + (0.5-0.7)*10/(0.3-0.7) = 15
        assert abs(profile.halftrace - 15.0) < 1e-9

    def test_perfect_shape_has_no_halftrace(self) -> None:
        profile = analyse_compliance({5: [1.0], 10: [1.0], 25: [1.0]})
        assert profile.shape == "perfect"
        assert profile.halftrace is None

    def test_bimodal_shape_has_no_halftrace(self) -> None:
        profile = analyse_compliance({5: [1.0, 1.0], 10: [0.0, 1.0]})
        assert profile.shape == "bimodal"
        assert profile.halftrace is None

    def test_gradient_without_crossing_returns_none(self) -> None:
        profile = analyse_compliance({5: [0.9], 10: [0.8], 20: [0.7]})
        assert profile.shape == "gradient"
        # Monotone but never crosses 0.5 → halftrace undefined
        assert profile.halftrace is None


class TestProfileFields:
    """Per-N means, variances, observation counts, and diagnostic flags."""

    def test_means_and_variances_recorded_per_n(self) -> None:
        profile = analyse_compliance({5: [0.0, 1.0], 10: [0.5, 0.5]})
        assert abs(profile.means_by_n[5] - 0.5) < 1e-9
        assert profile.means_by_n[10] == 0.5
        assert profile.variances_by_n[5] > 0
        assert profile.variances_by_n[10] == 0.0

    def test_n_values_sorted_ascending(self) -> None:
        profile = analyse_compliance({20: [1.0], 5: [1.0], 10: [1.0]})
        assert profile.n_values == [5, 10, 20]

    def test_n_observations_by_n_counts_reps(self) -> None:
        profile = analyse_compliance({5: [1.0, 1.0, 0.5], 10: [1.0]})
        assert profile.n_observations_by_n == {5: 3, 10: 1}

    def test_monotone_decreasing_flag(self) -> None:
        decreasing = analyse_compliance({5: [1.0], 10: [0.7], 20: [0.3]})
        assert decreasing.monotone_decreasing
        increasing = analyse_compliance({5: [0.3], 10: [0.7], 20: [1.0]})
        assert not increasing.monotone_decreasing

    def test_probe_name_is_threaded_through(self) -> None:
        profile = analyse_compliance({5: [1.0]}, probe="state_amnesia")
        assert profile.probe == "state_amnesia"


class TestValidation:
    """Bad input raises rather than silently producing nonsense."""

    def test_empty_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one N"):
            analyse_compliance({})

    def test_empty_reps_at_one_n_raises(self) -> None:
        with pytest.raises(ValueError, match="no observations"):
            analyse_compliance({5: [1.0], 10: []})

    def test_nan_score_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            analyse_compliance({5: [1.0, float("nan")]})

    def test_non_finite_commit_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="commit_threshold"):
            analyse_compliance({5: [1.0]}, commit_threshold=float("nan"))


class TestProfileSchema:
    """ComplianceProfile is a strict pydantic schema."""

    def test_profile_round_trips_through_json(self) -> None:
        profile = analyse_compliance({5: [1.0], 10: [0.7], 20: [0.3]})
        restored = ComplianceProfile.model_validate_json(profile.model_dump_json())
        assert restored == profile

    def test_profile_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ComplianceProfile.model_validate(
                {
                    "probe": "x",
                    "shape": "perfect",
                    "commit_probability": 1.0,
                    "n_values": [],
                    "means_by_n": {},
                    "variances_by_n": {},
                    "n_observations_by_n": {},
                    "max_within_n_variance": 0.0,
                    "monotone_decreasing": True,
                    "mystery": True,
                }
            )
