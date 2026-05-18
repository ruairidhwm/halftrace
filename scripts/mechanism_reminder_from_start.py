"""Does delivering the first reminder BEFORE strategy commitment rescue compliance?

`mechanism_periodic_reminder.py` at K=10 produced three discrete behaviours
across 9 reps at N=100 (Sonnet 4.6 STATUS counter rule):

    5/9 reps: perfect compliance
    2/9 reps: stable alternating compliance (score 0.505), insensitive to reminders
    2/9 reps: one-shot-rescue (score 0.149) — comply on the turn right after each
              reminder, ignore for 9 turns until the next one

The reminder rescued exactly one assistant turn per delivery for the
abandon-class reps and changed nothing at all for the alternation-class
reps. That falsifies attention-fade as the mechanism: a fading agent
would show graded compliance across d=0..9, not "100% at d=0, ~0%
everywhere else." The pattern that fits is **early-trajectory strategy
commitment** — the agent picks one of three compliance strategies in
the first ~10 turns and locks in. Reminders then briefly perturb the
locked strategy without changing it.

Diagnostic detail that motivates this script: both abandon-class reps
(rep 1 and rep 7) showed `1.1.1.1.1.` (alternating compliance) for the
first 10 turns — i.e. BEFORE the first reminder fires under the K=10
schedule. Then immediately after the first reminder lands at turn 11,
they collapsed to the one-shot pattern for the remainder. So they
spent the first 10 turns building a "this rule is optional" prior, and
the first reminder confirmed the wrong reading: "the user will remind
me when it matters; I don't need to do this on my own."

Hypothesis under test: if a reminder is visible from the very first
assistant turn (no opportunity to commit to a no-rule strategy), the
strategy commitment lands on full-compliance more often, and we see
fewer abandon-class reps. If compliance is unchanged from the original
K=10 condition, strategy commitment is robust to early intervention
and we need a different intervention class entirely (candidate B:
reminder inside tool_result content; candidate C: mid-trajectory
primer reinjection).

Implementation difference from `mechanism_periodic_reminder.py`: the
reminder is appended to the initial user message (visible to turn 1).
The K=10 cadence for subsequent reminders is preserved unchanged.
N=100, 10 reps, ~$3.
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

REMINDER_TEXT = (
    "Reminder: continue to start each text response with `STATUS: <N>` "
    "where <N> increments by 1 each response."
)


def run_with_reminder_from_start(
    task: Any,
    *,
    model: str,
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
) -> Trajectory:
    """Run task with reminder visible from turn 1, then K=10 cadence thereafter."""
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
    reminders_delivered = 1  # the initial-message reminder counts as #1

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
    out_path = Path(f"results/mechanism_reminder_start/{MODEL}_N{N}_k{K}.jsonl")
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
        f"running {len(plan)} reps on {MODEL} N={N} K={K} (reminder from turn 1)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=300.0)
    for rep in plan:
        print(f"  + from-start k={K} N={N} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(N, seed=rep, n_plants=1, discovery=False)
        trajectory = run_with_reminder_from_start(
            task, model=MODEL, client=client, max_iterations=500
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": N,
            "rep": rep,
            "k": K,
            "model": MODEL,
            "condition": f"reminder_from_start_k{K}",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
