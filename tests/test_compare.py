"""Tests for the diff-mode comparison helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from halftrace import (
    ComplianceProfile,
    ProbeComparison,
    analyse_compliance,
    compare_profiles,
)


def _profile(value: float, *, probe: str = "p") -> ComplianceProfile:
    """Helper: build a profile with a known commit_probability."""
    # commit_threshold is 0.95 by default; pick scores so commit_prob = `value`.
    # Two-element list lets us hit any commit_p in {0.0, 0.5, 1.0}.
    if value == 1.0:
        return analyse_compliance({5: [1.0, 1.0]}, probe=probe)
    if value == 0.5:
        return analyse_compliance({5: [1.0, 0.0]}, probe=probe)
    if value == 0.0:
        return analyse_compliance({5: [0.0, 0.0]}, probe=probe)
    raise ValueError(f"unsupported value {value}; pick 0.0 / 0.5 / 1.0")


class TestDirectionClassification:
    """Direction is decided by commit_probability delta vs threshold."""

    def test_large_positive_delta_is_improved(self) -> None:
        cmps = compare_profiles({"p": _profile(0.0)}, {"p": _profile(1.0)})
        assert cmps[0].direction == "improved"
        assert cmps[0].commit_probability_delta == 1.0

    def test_large_negative_delta_is_regressed(self) -> None:
        cmps = compare_profiles({"p": _profile(1.0)}, {"p": _profile(0.0)})
        assert cmps[0].direction == "regressed"
        assert cmps[0].commit_probability_delta == -1.0

    def test_no_delta_is_unchanged(self) -> None:
        cmps = compare_profiles({"p": _profile(1.0)}, {"p": _profile(1.0)})
        assert cmps[0].direction == "unchanged"
        assert cmps[0].commit_probability_delta == 0.0

    def test_threshold_governs_what_counts_as_a_change(self) -> None:
        # 0.5 vs 1.0 → delta = 0.5
        small_threshold = compare_profiles(
            {"p": _profile(0.5)},
            {"p": _profile(1.0)},
            delta_threshold=0.01,
        )
        assert small_threshold[0].direction == "improved"

        big_threshold = compare_profiles(
            {"p": _profile(0.5)},
            {"p": _profile(1.0)},
            delta_threshold=0.7,
        )
        assert big_threshold[0].direction == "unchanged"


class TestShapeChange:
    """`shape_changed` is informational and computed even within `unchanged`."""

    def test_shape_change_flag_when_shape_differs(self) -> None:
        before = analyse_compliance({5: [1.0, 1.0]})  # perfect
        # bimodal (high variance) but same commit_probability is hard to construct;
        # use two cells with different means
        after = analyse_compliance({5: [1.0, 0.0], 10: [1.0, 0.0]})
        cmps = compare_profiles({"p": before}, {"p": after})
        assert cmps[0].shape_changed is True

    def test_no_shape_change_flag_when_shape_same(self) -> None:
        cmps = compare_profiles({"p": _profile(1.0)}, {"p": _profile(1.0)})
        assert cmps[0].shape_changed is False


class TestPresenceAndAbsence:
    """Probes missing on one side are flagged distinctly."""

    def test_appeared_when_only_in_after(self) -> None:
        cmps = compare_profiles({}, {"p": _profile(0.5)})
        assert cmps[0].direction == "appeared"
        assert cmps[0].before is None
        assert cmps[0].after is not None

    def test_disappeared_when_only_in_before(self) -> None:
        cmps = compare_profiles({"p": _profile(0.5)}, {})
        assert cmps[0].direction == "disappeared"
        assert cmps[0].before is not None
        assert cmps[0].after is None

    def test_missing_when_both_are_none(self) -> None:
        cmps = compare_profiles({"p": None}, {"p": None})
        assert cmps[0].direction == "missing"
        assert cmps[0].commit_probability_delta is None


class TestProbeSetUnion:
    """Compare across the union of probes in both dicts."""

    def test_probes_only_in_after_are_included(self) -> None:
        cmps = compare_profiles(
            {"a": _profile(1.0)},
            {"a": _profile(1.0), "b": _profile(0.5)},
        )
        names = [c.probe for c in cmps]
        assert "a" in names and "b" in names

    def test_results_are_sorted_by_probe_name(self) -> None:
        cmps = compare_profiles(
            {"z": _profile(1.0), "a": _profile(1.0)},
            {"m": _profile(1.0)},
        )
        names = [c.probe for c in cmps]
        assert names == sorted(names)


class TestComparisonSchema:
    """ProbeComparison is a strict pydantic schema."""

    def test_round_trips_through_json(self) -> None:
        cmps = compare_profiles({"p": _profile(0.5)}, {"p": _profile(1.0)})
        original = cmps[0]
        restored = ProbeComparison.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ProbeComparison.model_validate(
                {
                    "probe": "x",
                    "before": None,
                    "after": None,
                    "direction": "missing",
                    "commit_probability_delta": None,
                    "shape_changed": False,
                    "mystery": True,
                }
            )
