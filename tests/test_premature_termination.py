"""Tests for the premature_termination probe."""

from __future__ import annotations

import pytest

from halftrace import premature_termination
from halftrace.trajectory import Trajectory


class TestPrematureTermination:
    def test_returns_none_when_no_annotation_present(self) -> None:
        t = Trajectory()
        t.add_turn("user", content="hi")
        t.add_turn("assistant", content="hello")
        score = premature_termination(t)
        assert score.value is None
        assert score.n_observations == 0

    def test_full_completion_scores_1(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 10, "n_completed": 10}})
        score = premature_termination(t)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_partial_completion_scores_ratio(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 10, "n_completed": 4}})
        score = premature_termination(t)
        assert score.value == 0.4

    def test_zero_completion_scores_0(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 10, "n_completed": 0}})
        score = premature_termination(t)
        assert score.value == 0.0

    def test_overcompletion_clamped_to_1(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 10, "n_completed": 12}})
        score = premature_termination(t)
        assert score.value == 1.0

    def test_turn_level_annotation_used_when_trajectory_metadata_absent(self) -> None:
        t = Trajectory()
        t.add_turn("user", content="task")
        t.add_turn("assistant", content="ok")
        t.add_turn(
            "tool",
            metadata={"premature_termination": {"n_required": 5, "n_completed": 3}},
        )
        score = premature_termination(t)
        assert score.value == 0.6
        assert score.n_observations == 1

    def test_trajectory_metadata_preferred_over_turn_metadata(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 10, "n_completed": 10}})
        t.add_turn(
            "tool",
            metadata={"premature_termination": {"n_required": 10, "n_completed": 0}},
        )
        score = premature_termination(t)
        assert score.value == 1.0

    def test_invalid_n_required_raises(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 0, "n_completed": 0}})
        with pytest.raises(ValueError, match="positive int"):
            premature_termination(t)

    def test_invalid_n_completed_raises(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 10, "n_completed": -1}})
        with pytest.raises(ValueError, match="non-negative int"):
            premature_termination(t)

    def test_malformed_metadata_raises(self) -> None:
        t = Trajectory(metadata={"premature_termination": "not a dict"})
        with pytest.raises(ValueError, match="must be a dict"):
            premature_termination(t)

    def test_details_includes_required_and_completed(self) -> None:
        t = Trajectory(metadata={"premature_termination": {"n_required": 7, "n_completed": 3}})
        score = premature_termination(t)
        assert score.details == {"n_required": 7, "n_completed": 3}
