"""Tests for the narration_substitution probe."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from halftrace import Score, ToolCall, Trajectory, narration_substitution


def _add_call_turn(
    traj: Trajectory,
    content: str | None,
    tools: list[tuple[str, dict[str, Any]]],
) -> None:
    tool_calls = [
        ToolCall(id=f"tc_{i}_{name}", name=name, args=args)
        for i, (name, args) in enumerate(tools)
    ]
    traj.add_turn(role="assistant", content=content, tool_calls=tool_calls)


def _add_text_only_turn(traj: Trajectory, content: str) -> None:
    traj.add_turn(role="assistant", content=content)


class TestEmptyOrTrivial:
    """No tool calls or no text turns return value=None."""

    def test_empty_trajectory_returns_none(self) -> None:
        score = narration_substitution(Trajectory())
        assert score.value is None
        assert score.n_observations == 0

    def test_no_tool_calls_anywhere_returns_none(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="hello")
        t.add_turn(role="assistant", content="hi there")
        assert narration_substitution(t).value is None

    def test_no_assistant_text_turns_returns_none(self) -> None:
        t = Trajectory()
        _add_call_turn(t, None, [("lookup", {"topic": "t1"})])
        assert narration_substitution(t).value is None


class TestScoring:
    """The substitution detection logic."""

    def test_narration_with_tool_call_is_not_substitution(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "Looking up topic_1 now.", [("lookup", {"topic": "t1"})])
        score = narration_substitution(t)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_text_only_turn_mentioning_tool_is_substitution(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "Looking up topic_1.", [("lookup", {"topic": "t1"})])
        _add_text_only_turn(t, "I'll lookup the next topic now.")
        score = narration_substitution(t)
        assert score.value == 0.5
        assert score.n_observations == 2
        assert score.details["n_substituting"] == 1

    def test_text_only_turn_not_mentioning_tool_is_not_substitution(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "Looking up.", [("lookup", {"topic": "t1"})])
        _add_text_only_turn(t, "All done; here is a summary of what I found.")
        # "summary" doesn't match "submit_summary" or "submit summary" exactly.
        # No mention of any tool name → not a substitution.
        score = narration_substitution(t)
        assert score.value == 1.0

    def test_underscore_to_space_variant_is_detected(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "first action", [("submit_summary", {"summary": "x"})])
        _add_text_only_turn(t, "Going to submit summary momentarily.")
        score = narration_substitution(t)
        assert score.value == 0.5

    def test_mixed_substitution_and_action(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "Looking up topic_1.", [("lookup", {"topic": "t1"})])
        _add_text_only_turn(t, "Going to lookup more.")
        _add_call_turn(t, "Looking up topic_2.", [("lookup", {"topic": "t2"})])
        _add_text_only_turn(t, "All done.")
        _add_text_only_turn(t, "Will lookup once more.")
        score = narration_substitution(t)
        assert score.value is not None
        assert abs(score.value - 3 / 5) < 1e-9
        assert score.n_observations == 5
        assert score.details["n_substituting"] == 2


class TestDetails:
    """Per-turn breakdown captures mentioned patterns."""

    def test_per_turn_records_mentioned_and_flag(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "Looking up topic_1.", [("lookup", {"topic": "t1"})])
        _add_text_only_turn(t, "I will lookup again later.")
        per_turn = narration_substitution(t).details["per_turn"]
        assert per_turn[0]["is_substitution"] is False
        assert per_turn[0]["has_tool_call"] is True
        assert per_turn[1]["is_substitution"] is True
        assert "lookup" in per_turn[1]["mentioned"]
        assert per_turn[1]["has_tool_call"] is False


class TestScoreSchema:
    """Round-trip and strict schema."""

    def test_score_round_trips(self) -> None:
        t = Trajectory()
        _add_call_turn(t, "Looking up t1.", [("lookup", {"topic": "t1"})])
        score = narration_substitution(t)
        restored = Score.model_validate_json(score.model_dump_json())
        assert restored == score

    def test_score_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Score.model_validate(
                {"probe": "narration_substitution", "value": 1.0, "mystery": True}
            )
