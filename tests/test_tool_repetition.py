"""Tests for the tool_repetition probe."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from halftrace import Score, ToolCall, Trajectory, tool_repetition


def _add_call(
    traj: Trajectory,
    name: str,
    args: dict[str, Any] | None = None,
    id_: str | None = None,
) -> None:
    args_dict: dict[str, Any] = args if args is not None else {}
    call_id: str = id_ if id_ is not None else f"tc_{len(traj.tool_calls())}"
    traj.add_turn(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name=name, args=args_dict)],
    )


class TestEmptyOrTrivial:
    """Trajectories with no tool calls return value=None."""

    def test_empty_trajectory_returns_none(self) -> None:
        score = tool_repetition(Trajectory())
        assert score.value is None
        assert score.n_observations == 0

    def test_trajectory_with_no_tool_calls_returns_none(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="hello")
        t.add_turn(role="assistant", content="hi")
        assert tool_repetition(t).value is None


class TestScoring:
    """The core uniqueness logic."""

    def test_single_call_scores_one(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})
        score = tool_repetition(t)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_all_unique_calls_score_one(self) -> None:
        t = Trajectory()
        for i in range(5):
            _add_call(t, "lookup", {"topic": f"t{i}"})
        score = tool_repetition(t)
        assert score.value == 1.0
        assert score.n_observations == 5

    def test_exact_repeat_is_a_duplicate(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})
        _add_call(t, "lookup", {"topic": "t1"})
        score = tool_repetition(t)
        assert score.value == 0.5
        assert score.n_observations == 2
        assert score.details["n_duplicates"] == 1

    def test_same_name_different_args_is_not_a_duplicate(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})
        _add_call(t, "lookup", {"topic": "t2"})
        assert tool_repetition(t).value == 1.0

    def test_different_key_order_still_counts_as_duplicate(self) -> None:
        t = Trajectory()
        _add_call(t, "tool", {"a": 1, "b": 2})
        _add_call(t, "tool", {"b": 2, "a": 1})
        assert tool_repetition(t).value == 0.5

    def test_mixed_unique_and_duplicate_calls(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})  # unique
        _add_call(t, "lookup", {"topic": "t2"})  # unique
        _add_call(t, "lookup", {"topic": "t1"})  # duplicate
        _add_call(t, "submit_summary", {"summary": "x"})  # unique
        _add_call(t, "lookup", {"topic": "t2"})  # duplicate
        score = tool_repetition(t)
        assert score.value is not None
        assert abs(score.value - 3 / 5) < 1e-9
        assert score.n_observations == 5
        assert score.details["n_duplicates"] == 2

    def test_three_calls_to_same_target_yield_two_duplicates(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})
        _add_call(t, "lookup", {"topic": "t1"})
        _add_call(t, "lookup", {"topic": "t1"})
        score = tool_repetition(t)
        assert score.value is not None
        assert abs(score.value - 1 / 3) < 1e-9
        assert score.details["n_duplicates"] == 2


class TestDetails:
    """Per-call breakdown is recorded for diagnosis."""

    def test_per_call_marks_each_entry(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})
        _add_call(t, "lookup", {"topic": "t1"})
        per_call = tool_repetition(t).details["per_call"]
        assert per_call[0]["is_duplicate"] is False
        assert per_call[1]["is_duplicate"] is True
        assert per_call[0]["name"] == "lookup"

    def test_per_call_records_turn_index(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="hi")
        _add_call(t, "lookup", {"topic": "t1"})
        per_call = tool_repetition(t).details["per_call"]
        assert per_call[0]["turn_index"] == 1


class TestScoreSchema:
    """Score serializes and rejects extras."""

    def test_score_round_trips(self) -> None:
        t = Trajectory()
        _add_call(t, "lookup", {"topic": "t1"})
        _add_call(t, "lookup", {"topic": "t1"})
        score = tool_repetition(t)
        restored = Score.model_validate_json(score.model_dump_json())
        assert restored == score

    def test_score_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Score.model_validate(
                {"probe": "tool_repetition", "value": 0.5, "mystery": True}
            )
