"""Tests for the instruction_decay probe."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from halftrace import Score, Trajectory, instruction_decay


def _trajectory_with_rule(
    rule_id: str = "end_with_marker",
    params: dict[str, str] | None = None,
) -> Trajectory:
    t = Trajectory()
    t.metadata["instruction_decay"] = {
        "rule_id": rule_id,
        "params": params if params is not None else {"marker": "[[END]]"},
    }
    return t


class TestEmptyOrTrivial:
    """Trajectories with no rule or no observations return value=None."""

    def test_no_rule_returns_none(self) -> None:
        t = Trajectory()
        t.add_turn(role="assistant", content="hello [[END]]")
        assert instruction_decay(t).value is None

    def test_rule_but_no_assistant_text_returns_none(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="user", content="hi")
        # Assistant turn with no content (tool-only)
        t.add_turn(role="assistant")
        score = instruction_decay(t)
        assert score.value is None
        assert score.n_observations == 0


class TestEndWithMarker:
    """The built-in end_with_marker rule."""

    def test_single_turn_following_rule_scores_one(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="assistant", content="Looking up topic_1. [[END]]")
        score = instruction_decay(t)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_single_turn_violating_rule_scores_zero(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="assistant", content="Looking up topic_1.")
        score = instruction_decay(t)
        assert score.value == 0.0

    def test_partial_compliance_scores_fraction(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="assistant", content="Step 1. [[END]]")
        t.add_turn(role="assistant", content="Step 2.")  # missing marker
        t.add_turn(role="assistant", content="Step 3. [[END]]")
        t.add_turn(role="assistant", content="Step 4.")  # missing marker
        score = instruction_decay(t)
        assert score.value == 0.5
        assert score.n_observations == 4

    def test_trailing_whitespace_after_marker_still_counts(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="assistant", content="Done. [[END]]   \n")
        assert instruction_decay(t).value == 1.0

    def test_trailing_punctuation_after_marker_fails_rule(self) -> None:
        # Deliberate: the rule is strict end-with, no trailing punctuation.
        t = _trajectory_with_rule()
        t.add_turn(role="assistant", content="Done. [[END]].")
        assert instruction_decay(t).value == 0.0


class TestRoleFiltering:
    """Only assistant turns with text content are scored."""

    def test_user_and_tool_turns_are_ignored(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="user", content="not scored [[END]]")
        t.add_turn(role="tool", content="also not scored [[END]]")
        t.add_turn(role="assistant", content="this one [[END]]")
        score = instruction_decay(t)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_assistant_turns_without_content_are_skipped(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="assistant")  # no content
        t.add_turn(role="assistant", content="text [[END]]")
        score = instruction_decay(t)
        assert score.value == 1.0
        assert score.n_observations == 1


class TestCustomMatcher:
    """The matcher hook lets callers score arbitrary rules."""

    def test_custom_matcher_replaces_builtin(self) -> None:
        t = _trajectory_with_rule(
            rule_id="anything", params={"forbidden": "secret"}
        )
        t.add_turn(role="assistant", content="this contains secret")
        t.add_turn(role="assistant", content="this does not")

        def no_forbidden(content: str, params: dict[str, str]) -> bool:
            return params["forbidden"] not in content

        score = instruction_decay(t, matcher=no_forbidden)
        assert score.value == 0.5

    def test_custom_matcher_is_called_with_params(self) -> None:
        seen: list[tuple[str, dict[str, str]]] = []
        t = _trajectory_with_rule(params={"marker": "<<X>>"})
        t.add_turn(role="assistant", content="hello")

        def collect(content: str, params: dict[str, str]) -> bool:
            seen.append((content, params))
            return True

        instruction_decay(t, matcher=collect)
        assert seen == [("hello", {"marker": "<<X>>"})]


class TestValidation:
    """Malformed metadata raises loudly."""

    def test_non_dict_metadata_raises(self) -> None:
        t = Trajectory()
        t.metadata["instruction_decay"] = "not a dict"
        with pytest.raises(ValueError, match="must be a dict"):
            instruction_decay(t)

    def test_unknown_rule_id_raises(self) -> None:
        t = _trajectory_with_rule(rule_id="not-a-real-rule")
        t.add_turn(role="assistant", content="text")
        with pytest.raises(ValueError, match="unknown rule_id"):
            instruction_decay(t)

    def test_non_string_rule_id_raises(self) -> None:
        t = Trajectory()
        t.metadata["instruction_decay"] = {"rule_id": 42, "params": {}}
        t.add_turn(role="assistant", content="text")
        with pytest.raises(ValueError, match="rule_id"):
            instruction_decay(t)

    def test_non_dict_params_raises(self) -> None:
        t = Trajectory()
        t.metadata["instruction_decay"] = {
            "rule_id": "end_with_marker",
            "params": "not a dict",
        }
        with pytest.raises(ValueError, match="params"):
            instruction_decay(t)

    def test_end_with_marker_without_marker_param_raises(self) -> None:
        t = _trajectory_with_rule(params={})
        t.add_turn(role="assistant", content="text")
        with pytest.raises(ValueError, match="marker"):
            instruction_decay(t)


class TestScoreSchema:
    """The Score model round-trips and rejects extras."""

    def test_score_round_trips(self) -> None:
        t = _trajectory_with_rule()
        t.add_turn(role="assistant", content="hi [[END]]")
        score = instruction_decay(t)
        restored = Score.model_validate_json(score.model_dump_json())
        assert restored == score

    def test_score_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Score.model_validate(
                {"probe": "instruction_decay", "value": 1.0, "mystery": True}
            )
