"""Tests for the state_amnesia probe."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from halftrace import Score, ToolCall, ToolResult, Trajectory, state_amnesia

_FLOAT_TOL = 1e-9


def _plant(traj: Trajectory, fact_id: str, fact: str, content: str | None = None) -> None:
    traj.add_turn(
        role="user",
        content=content if content is not None else f"Remember: {fact}",
        metadata={"state_amnesia": {"role": "plant", "fact_id": fact_id, "fact": fact}},
    )


def _recall(traj: Trajectory, fact_id: str, content: str = "What was it?") -> None:
    traj.add_turn(
        role="user",
        content=content,
        metadata={"state_amnesia": {"role": "recall", "fact_id": fact_id}},
    )


class TestEmptyOrTrivial:
    """Trajectories with nothing to score return value=None."""

    def test_empty_trajectory_returns_none_value(self) -> None:
        score = state_amnesia(Trajectory())
        assert score.value is None
        assert score.n_observations == 0
        assert score.probe == "state_amnesia"

    def test_trajectory_with_only_plants_returns_none(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        score = state_amnesia(t)
        assert score.value is None
        assert score.n_observations == 0

    def test_trajectory_with_unannotated_turns_returns_none(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="hello")
        t.add_turn(role="assistant", content="hi")
        score = state_amnesia(t)
        assert score.value is None


class TestScoring:
    """The core scoring path: plants → recalls → assistant response."""

    def test_single_correct_recall_scores_one(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="the door is red, of course")
        score = state_amnesia(t)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_single_incorrect_recall_scores_zero(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="I have no idea.")
        score = state_amnesia(t)
        assert score.value == 0.0
        assert score.n_observations == 1

    def test_mixed_recalls_score_a_fraction(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _plant(t, "f2", "the cat is named Mittens")
        _plant(t, "f3", "the year is 1987")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="the door is red")
        _recall(t, "f2")
        t.add_turn(role="assistant", content="I don't recall")
        _recall(t, "f3")
        t.add_turn(role="assistant", content="the year is 1987")
        score = state_amnesia(t)
        assert score.value is not None
        assert abs(score.value - 2 / 3) < _FLOAT_TOL
        assert score.n_observations == 3

    def test_details_record_per_fact_outcomes(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _plant(t, "f2", "the cat is grey")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="the door is red")
        _recall(t, "f2")
        t.add_turn(role="assistant", content="no clue")
        score = state_amnesia(t)
        per_fact = score.details["per_fact"]
        assert per_fact[0] == {"fact_id": "f1", "correct": True}
        assert per_fact[1] == {"fact_id": "f2", "correct": False}


class TestMatcher:
    """The default matcher and the matcher injection point."""

    def test_default_matcher_is_case_insensitive(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "The Door Is Red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="THE DOOR IS RED")
        assert state_amnesia(t).value == 1.0

    def test_default_matcher_normalises_whitespace(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the   door  is\tred")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="the door is red")
        assert state_amnesia(t).value == 1.0

    def test_default_matcher_requires_full_phrase(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="something about a door")
        assert state_amnesia(t).value == 0.0

    def test_custom_matcher_can_be_supplied(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="?")
        score = state_amnesia(t, matcher=lambda planted, response: True)
        assert score.value == 1.0

    def test_custom_matcher_receives_planted_and_response(self) -> None:
        seen: list[tuple[str, str]] = []

        def matcher(planted: str, response: str) -> bool:
            seen.append((planted, response))
            return True

        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="ok")
        state_amnesia(t, matcher=matcher)
        assert seen == [("the door is red", "ok")]


class TestResponseLookup:
    """How the probe locates the assistant response after a recall."""

    def test_intervening_tool_turns_do_not_block_lookup(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(
            role="assistant",
            tool_calls=[ToolCall(id="t1", name="lookup", args={})],
        )
        t.add_turn(
            role="tool",
            tool_results=[ToolResult(tool_call_id="t1", name="lookup", result="...")],
        )
        t.add_turn(role="assistant", content="the door is red")
        assert state_amnesia(t).value == 1.0

    def test_no_assistant_response_after_recall_counts_as_wrong(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        score = state_amnesia(t)
        assert score.value == 0.0
        assert score.n_observations == 1
        assert score.details["per_fact"][0]["reason"] == "no_response"

    def test_first_text_assistant_turn_is_used_not_a_later_one(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="I don't know")
        t.add_turn(role="assistant", content="actually, the door is red")
        # The first text turn wins, so this is scored as incorrect.
        assert state_amnesia(t).value == 0.0


class TestValidation:
    """Malformed annotations raise loudly rather than silently scoring zero."""

    def test_recall_without_plant_raises(self) -> None:
        t = Trajectory()
        _recall(t, "f_unknown")
        t.add_turn(role="assistant", content="?")
        with pytest.raises(ValueError, match="unplanted"):
            state_amnesia(t)

    def test_non_dict_annotation_raises(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", metadata={"state_amnesia": "not a dict"})
        with pytest.raises(ValueError, match="must be a dict"):
            state_amnesia(t)

    def test_missing_fact_id_raises(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", metadata={"state_amnesia": {"role": "plant", "fact": "x"}})
        with pytest.raises(ValueError, match="fact_id"):
            state_amnesia(t)

    def test_plant_without_fact_text_raises(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", metadata={"state_amnesia": {"role": "plant", "fact_id": "f1"}})
        with pytest.raises(ValueError, match="missing a string fact"):
            state_amnesia(t)

    def test_unknown_role_raises(self) -> None:
        t = Trajectory()
        t.add_turn(
            role="user",
            metadata={"state_amnesia": {"role": "weird", "fact_id": "f1"}},
        )
        with pytest.raises(ValueError, match="unknown role"):
            state_amnesia(t)


class TestScoreSchema:
    """The Score model is a real pydantic schema with strict fields."""

    def test_score_round_trips_through_json(self) -> None:
        t = Trajectory()
        _plant(t, "f1", "the door is red")
        _recall(t, "f1")
        t.add_turn(role="assistant", content="the door is red")
        score = state_amnesia(t)
        restored = Score.model_validate_json(score.model_dump_json())
        assert restored == score

    def test_score_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Score.model_validate({"probe": "x", "value": 1.0, "mystery": True})

    def test_score_value_can_be_none(self) -> None:
        s = Score(probe="state_amnesia", value=None)
        assert s.value is None
        assert s.n_observations == 0
