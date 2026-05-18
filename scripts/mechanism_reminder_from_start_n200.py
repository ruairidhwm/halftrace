"""Reminder-from-start at N=200 — does the 10/10 rescue hold at double depth?

`mechanism_reminder_from_start.py` at N=100 produced 10/10 perfect on
Sonnet 4.6's STATUS counter rule (mean 1.000), confirming the strategy-
lock-in mechanism: a reminder in the initial user message pre-empts the
commitment-to-non-compliance that drives baseline abandonment.

That result is at N=100, where baseline compliance is already coinflip
(mean 0.51). The harder test is N=200, where baseline compliance is
mean 0.16 (~1/10 commit). If reminder-from-start still drives compliance
to 10/10 at N=200, the rescue is robust at twice the depth and the
practical recommendation ("restate the rule in the initial user
message") generalises to long-running agents.

If it doesn't — say 6/10 or 7/10 — then strategy lock-in is the
dominant mechanism at N=100 but a secondary mechanism kicks in at
N=200 that the early reminder doesn't address. That would tell us the
rescue has a depth ceiling and we need a different intervention for
the truly long trajectory regime.

Implementation: identical to `mechanism_reminder_from_start.py` except
N=200 and output path differs. Same K=10 cadence, same reminder text,
same initial-user-message placement.

10 reps. ~$5 (caching helps; the linear cost growth with N is partly
offset by Sonnet's prompt caching at long context).
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
N = 200
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
    max_iterations: int = 600,
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
    out_path = Path(
        f"results/mechanism_reminder_start/{MODEL}_N{N}_k{K}.jsonl"
    )
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
    client = anthropic.Anthropic(max_retries=20, timeout=600.0)
    for rep in plan:
        print(f"  + from-start k={K} N={N} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(N, seed=rep, n_plants=1, discovery=False)
        trajectory = run_with_reminder_from_start(
            task, model=MODEL, client=client, max_iterations=600
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": N,
            "rep": rep,
            "k": K,
            "model": MODEL,
            "condition": f"reminder_from_start_k{K}_n{N}",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
