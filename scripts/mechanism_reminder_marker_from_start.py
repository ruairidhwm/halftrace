"""Reminder-from-start on the marker rule — does the rescue generalise across rule families?

`mechanism_reminder_from_start.py` at N=100 produced 10/10 perfect on
Sonnet 4.6's STATUS counter rule (mean 1.000) by placing a rule
reminder in the initial user message. The mechanism story:
strategy-lock-in by turn ~10, pre-empted by an in-trajectory reminder
before commitment forms.

The counter rule is one rule on one model. The marker rule
(`mechanism_marker_rule.py`) — "end every text response with the
literal marker `[[END]]` on its own" — differs along two axes:
prefix vs suffix placement, and stateful (incrementing counter) vs
stateless (constant marker). It also showed non-monotone behaviour
under the cold-start cure: baseline 0.79 / zero 0.58 (WORSE) / minimal
0.79 / full 0.97. If the marker rule shares the strategy-lock-in
mechanism on Sonnet, reminder-from-start should still rescue it at
N=100 — possibly to 10/10, possibly to a high-but-imperfect rate
given the rule's already-different baseline.

Three possible outcomes:

    10/10 perfect:    strategy lock-in is rule-family-general on Sonnet.
                      "Restate in initial user message" is the universal
                      practical recommendation for this model and any
                      structurally similar rule.
    8-9/10 perfect:   intervention is robust but not universal; the
                      marker rule has a residual failure mode the
                      counter rule didn't have.
    Substantially worse: rule design interacts with the mechanism.
                      Strategy lock-in is real for counter rules but
                      something else drives marker-rule abandonment.
                      Would need a different intervention for marker-
                      style rules.

Implementation: same agent loop as `mechanism_reminder_from_start.py`,
but with the task's system prompt rewritten to the marker rule (same
override helper logic as `mechanism_marker_rule.py`) and the reminder
text changed to match.

N=100, 10 reps. ~$3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import anthropic
from _primer_runner import score_all
from anthropic.types import Message, Usage

from halftrace.adapters.anthropic_adapter import (
    _accumulate_usage,
    _parse_response,
    _to_anthropic_tools,
    _to_param_blocks,
)
from halftrace.tasks import find_and_synthesise
from halftrace.trajectory import ToolResult, Trajectory

MODEL = "claude-sonnet-4-6"
N = 100
REPS = 10
K = 10
MARKER = "[[END]]"

MARKER_RULE_PROMPT_HEADER = (
    "You are a research assistant.\n\n"
    f"OUTPUT FORMAT (must be followed every single turn): end every text "
    f"response you produce with the literal marker `{MARKER}` on its own. "
    "The marker must be the very last token of your reply.\n\n"
    "Example of a correctly formatted assistant response:\n"
    f"  Looking up topic_3 now. {MARKER}\n\n"
)

REMINDER_TEXT = (
    f"Reminder: continue to end each text response with the literal marker "
    f"`{MARKER}` on its own line as the very last token."
)


def override_task_with_marker_rule(task: Any) -> None:
    """Mutate find_and_synthesise to use the end-with-marker rule.

    Mirrors `mechanism_marker_rule.override_task_with_marker_rule` so
    the experiments are directly comparable.
    """
    base_prompt = task.system_prompt
    task_section_start = base_prompt.find("TASK:")
    if task_section_start == -1:
        raise RuntimeError("Could not find 'TASK:' header in task system_prompt")
    task_section = base_prompt[task_section_start:]
    task.system_prompt = MARKER_RULE_PROMPT_HEADER + task_section
    task.trajectory_metadata["instruction_decay"] = {
        "rule_id": "end_with_marker",
        "params": {"marker": MARKER},
    }


def run_with_reminder_from_start(
    task: Any,
    *,
    model: str,
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
) -> Trajectory:
    """Run task with reminder visible from turn 1, then K cadence thereafter."""
    initial_user_message = task.initial_user_message + "\n\n" + REMINDER_TEXT

    trajectory = Trajectory(task_id=task.id, model=model)
    for key, value in task.trajectory_metadata.items():
        trajectory.metadata[key] = value
    trajectory.add_turn("system", content=task.system_prompt)
    trajectory.add_turn("user", content=initial_user_message)

    anthropic_tools = _to_anthropic_tools(task.tool_specs, cache=True)
    system_param: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": task.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_user_message},
    ]
    usage_totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "n_requests": 0,
    }
    reminders_delivered = 1

    for i in range(max_iterations):
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": messages,
            "tools": anthropic_tools,
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": True,
            },
        }
        response = cast("Message", client.messages.create(**create_kwargs))
        _accumulate_usage(usage_totals, cast("Usage", response.usage))
        text_content, tool_calls, tool_uses = _parse_response(response)

        trajectory.add_turn(
            "assistant",
            content=text_content,
            tool_calls=tool_calls,
            metadata={"stop_reason": response.stop_reason},
        )

        if not tool_uses:
            break

        messages.append(
            {"role": "assistant", "content": _to_param_blocks(response.content)}
        )

        tool_result_blocks: list[dict[str, Any]] = []
        for tu_id, tu_name, tu_input in tool_uses:
            task_response = task.handle_tool_call(tu_name, tu_input)
            trajectory.add_turn(
                "tool",
                tool_results=[
                    ToolResult(
                        tool_call_id=tu_id,
                        name=tu_name,
                        result=task_response.result,
                        is_error=task_response.is_error,
                    )
                ],
                metadata=task_response.annotations,
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": task_response.result,
                    "is_error": task_response.is_error,
                }
            )

        user_content: list[dict[str, Any]] = list(tool_result_blocks)
        if (i + 1) % K == 0:
            user_content.append({"type": "text", "text": REMINDER_TEXT})
            reminders_delivered += 1
        messages.append({"role": "user", "content": user_content})

        if task.is_done():
            break

    trajectory.metadata["usage_totals"] = usage_totals
    trajectory.metadata["reminder_k"] = K
    trajectory.metadata["reminder_text"] = REMINDER_TEXT
    trajectory.metadata["reminder_from_start"] = True
    trajectory.metadata["reminders_delivered"] = reminders_delivered
    return trajectory


def main() -> int:
    out_path = Path(f"results/mechanism_reminder_start/{MODEL}_N{N}_k{K}_marker.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(int(r["rep"]))

    plan = [rep for rep in range(REPS) if rep not in done]
    if not plan:
        print("all reps done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} reps on {MODEL} N={N} K={K} "
        "(marker rule, reminder from turn 1)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=300.0)
    for rep in plan:
        print(
            f"  + marker from-start k={K} N={N} rep={rep}",
            file=sys.stderr,
            flush=True,
        )
        task = find_and_synthesise(N, seed=rep, n_plants=1, discovery=False)
        override_task_with_marker_rule(task)
        trajectory = run_with_reminder_from_start(
            task, model=MODEL, client=client, max_iterations=500
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": N,
            "rep": rep,
            "k": K,
            "model": MODEL,
            "rule": f"end_with_marker/{MARKER}",
            "condition": f"reminder_from_start_k{K}_marker",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
