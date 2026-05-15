"""Tests for trajectory ingestion from Anthropic and OpenAI message formats."""

from __future__ import annotations

from typing import Any

from halftrace import from_anthropic_messages, from_openai_messages


class TestAnthropicIngestion:
    """Convert Anthropic messages.create() payloads."""

    def test_string_system_prompt_becomes_system_turn(self) -> None:
        payload: dict[str, Any] = {
            "system": "you are a helpful assistant",
            "messages": [],
        }
        t = from_anthropic_messages(payload)
        assert len(t.turns) == 1
        assert t.turns[0].role == "system"
        assert t.turns[0].content == "you are a helpful assistant"

    def test_list_system_blocks_are_concatenated(self) -> None:
        payload: dict[str, Any] = {
            "system": [
                {"type": "text", "text": "part one"},
                {"type": "text", "text": "part two"},
            ],
            "messages": [],
        }
        t = from_anthropic_messages(payload)
        assert t.turns[0].content == "part one\npart two"

    def test_string_user_content_becomes_user_turn(self) -> None:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": "hello"}],
        }
        t = from_anthropic_messages(payload)
        assert [turn.role for turn in t.turns] == ["user"]
        assert t.turns[0].content == "hello"

    def test_assistant_text_only_becomes_assistant_turn(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello back"},
            ],
        }
        t = from_anthropic_messages(payload)
        assert t.turns[1].role == "assistant"
        assert t.turns[1].content == "hello back"

    def test_assistant_tool_use_becomes_tool_call(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "looking up"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"topic": "t1"},
                        },
                    ],
                },
            ],
        }
        t = from_anthropic_messages(payload)
        assert t.turns[0].role == "assistant"
        assert t.turns[0].content == "looking up"
        assert len(t.turns[0].tool_calls) == 1
        tc = t.turns[0].tool_calls[0]
        assert tc.id == "toolu_1"
        assert tc.name == "lookup"
        assert tc.args == {"topic": "t1"}

    def test_user_tool_result_becomes_tool_turn(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "the result",
                        }
                    ],
                },
            ],
        }
        t = from_anthropic_messages(payload)
        roles = [turn.role for turn in t.turns]
        assert roles == ["assistant", "tool"]
        tool_turn = t.turns[1]
        assert len(tool_turn.tool_results) == 1
        tr = tool_turn.tool_results[0]
        assert tr.tool_call_id == "toolu_1"
        assert tr.name == "lookup"  # back-filled from the prior assistant tool_use
        assert tr.result == "the result"

    def test_tool_result_error_flag_is_preserved(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                },
            ],
        }
        t = from_anthropic_messages(payload)
        assert t.turns[0].tool_results[0].is_error

    def test_tool_result_content_list_is_joined(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "line one"},
                                {"type": "text", "text": "line two"},
                            ],
                        }
                    ],
                },
            ],
        }
        t = from_anthropic_messages(payload)
        assert t.turns[0].tool_results[0].result == "line one\nline two"

    def test_metadata_and_model_are_threaded_through(self) -> None:
        payload: dict[str, Any] = {
            "model": "claude-sonnet-4-6",
            "task_id": "my-task",
            "metadata": {
                "instruction_decay": {
                    "rule_id": "end_with_marker",
                    "params": {"marker": "[[END]]"},
                }
            },
            "messages": [],
        }
        t = from_anthropic_messages(payload)
        assert t.model == "claude-sonnet-4-6"
        assert t.task_id == "my-task"
        assert "instruction_decay" in t.metadata


class TestOpenAIIngestion:
    """Convert OpenAI chat completions payloads."""

    def test_system_message_becomes_system_turn(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "system prompt"},
            ],
        }
        t = from_openai_messages(payload)
        assert t.turns[0].role == "system"
        assert t.turns[0].content == "system prompt"

    def test_user_and_assistant_round_trip(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        t = from_openai_messages(payload)
        assert [turn.role for turn in t.turns] == ["user", "assistant"]
        assert t.turns[0].content == "hi"
        assert t.turns[1].content == "hello"

    def test_assistant_tool_calls_have_args_parsed_from_json_string(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "looking up topic_1",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"topic": "t1"}',
                            },
                        }
                    ],
                },
            ],
        }
        t = from_openai_messages(payload)
        tc = t.turns[0].tool_calls[0]
        assert tc.id == "call_1"
        assert tc.name == "lookup"
        assert tc.args == {"topic": "t1"}

    def test_malformed_arguments_string_becomes_empty_dict(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": "{not json"},
                        }
                    ],
                },
            ],
        }
        t = from_openai_messages(payload)
        assert t.turns[0].tool_calls[0].args == {}

    def test_tool_role_message_becomes_tool_turn_with_back_filled_name(self) -> None:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
        }
        t = from_openai_messages(payload)
        assert [turn.role for turn in t.turns] == ["assistant", "tool"]
        tr = t.turns[1].tool_results[0]
        assert tr.tool_call_id == "call_1"
        assert tr.name == "lookup"
        assert tr.result == "result"


class TestRoundTripCompatibility:
    """An ingested trajectory should work with the existing probes."""

    def test_ingested_trajectory_can_be_scored_by_tool_repetition(self) -> None:
        from halftrace import tool_repetition

        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "1", "function": {"name": "lookup", "arguments": '{"x":1}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "1", "content": "ok"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "2", "function": {"name": "lookup", "arguments": '{"x":1}'}},
                    ],
                },
            ],
        }
        t = from_openai_messages(payload)
        score = tool_repetition(t)
        # Two calls with the same (name, args) — second is a duplicate
        assert score.value == 0.5

    def test_ingested_trajectory_with_metadata_can_be_scored_by_instruction_decay(self) -> None:
        from halftrace import instruction_decay

        payload: dict[str, Any] = {
            "metadata": {
                "instruction_decay": {
                    "rule_id": "end_with_marker",
                    "params": {"marker": "[[END]]"},
                }
            },
            "messages": [
                {"role": "assistant", "content": "ack [[END]]"},
                {"role": "assistant", "content": "no marker here"},
            ],
        }
        t = from_openai_messages(payload)
        score = instruction_decay(t)
        assert score.value == 0.5
        assert score.n_observations == 2
