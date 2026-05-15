"""Tests for the shape-to-recommendation catalog."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from halftrace import (
    ComplianceProfile,
    Diagnosis,
    analyse_compliance,
    diagnose,
)


def _profile(shape: str = "perfect") -> ComplianceProfile:
    """Build a real profile with the requested shape via analyse_compliance."""
    if shape == "perfect":
        return analyse_compliance({5: [1.0, 1.0], 10: [1.0]})
    if shape == "abandoned":
        return analyse_compliance({5: [0.0, 0.0], 10: [0.0]})
    if shape == "bimodal":
        return analyse_compliance({5: [1.0, 1.0], 10: [0.0, 1.0]})
    if shape == "categorical":
        return analyse_compliance({10: [0.55, 0.55], 25: [0.52, 0.52]})
    if shape == "gradient":
        return analyse_compliance({5: [1.0], 10: [0.7], 20: [0.3]})
    raise ValueError(f"unknown shape: {shape}")


class TestDiagnoseEverySupportedShape:
    """The diagnose() function returns a useful Diagnosis for every shape."""

    @pytest.mark.parametrize(
        "shape", ["perfect", "abandoned", "bimodal", "categorical", "gradient"]
    )
    def test_diagnosis_is_returned_for_every_shape(self, shape: str) -> None:
        profile = _profile(shape)
        assert profile.shape == shape
        d = diagnose(profile)
        assert d.shape == shape
        assert d.cause
        assert d.suggestions
        # Every suggestion is a non-empty string
        for s in d.suggestions:
            assert isinstance(s, str)
            assert s.strip()


class TestDiagnosisContent:
    """Spot-check that diagnoses actually reflect what we know from the pilots."""

    def test_perfect_says_no_action_needed(self) -> None:
        d = diagnose(_profile("perfect"))
        assert any("no action" in s.lower() for s in d.suggestions)

    def test_bimodal_mentions_worked_example_or_first_turn(self) -> None:
        d = diagnose(_profile("bimodal"))
        combined = " ".join(d.suggestions).lower()
        # The pilot finding: adding a worked example or relaxing turn 1
        # increases commit probability
        assert "example" in combined or "first" in combined

    def test_categorical_mentions_turn_type_scope(self) -> None:
        d = diagnose(_profile("categorical"))
        combined = (d.cause + " ".join(d.suggestions)).lower()
        # Categorical's signature is turn-type discrimination
        assert "turn-type" in combined or "turn type" in combined or "per_turn" in combined

    def test_abandoned_suggests_moving_rule_or_simplifying(self) -> None:
        d = diagnose(_profile("abandoned"))
        combined = " ".join(d.suggestions).lower()
        # The pilot finding: buried rules get dropped from turn 1
        assert "first line" in combined or "simpler" in combined or "example" in combined


class TestDiagnosisSchema:
    """The Diagnosis pydantic model is strict and round-trips."""

    def test_diagnosis_round_trips_through_json(self) -> None:
        d = diagnose(_profile("bimodal"))
        restored = Diagnosis.model_validate_json(d.model_dump_json())
        assert restored == d

    def test_diagnosis_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Diagnosis.model_validate(
                {
                    "shape": "perfect",
                    "cause": "x",
                    "suggestions": [],
                    "mystery": True,
                }
            )
