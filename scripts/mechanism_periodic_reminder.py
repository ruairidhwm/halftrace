"""Does periodic rule restatement rescue Sonnet 4.6's long-N compliance drift?

Sonnet 4.6 on the STATUS counter rule (no primer):

    N=10   mean 0.345  bimodal (1/5 commit)
    N=100  mean 0.51   (~5/10 commit)
    N=200  mean 0.16   (~1/10 commit)

The two-trivial-exchange cold-start primer (`mechanism_control.py`) lifts
N=10 to mean 1.00 but does NOT rescue N=100 (`mechanism_long.py` shows
mean 0.47, indistinguishable from baseline). So whatever drives long-N
abandonment is a different mechanism — it kicks in mid-trajectory after
the agent has already been complying.

This script tests the dominant candidate: **system-prompt attention
fade**. Hypothesis: by turn 50-ish the rule, declared once at the very
top of context, has lost enough salience that the agent stops applying
it. If we periodically re-surface the rule in user-side content (where
the latest turns get the most attention), compliance should hold.

The intervention: append a short reminder text block to the user-side
message that carries every K-th tool_result. The reminder is a tight
paraphrase of the system-prompt rule — same operative content, ~30
tokens rather than ~80. Two K values give a dose-response:

    k=10: reminder before turns 11, 21, ..., 91  (9 reminders / 100 turns)
    k=25: reminder before turns 26, 51, 76       (3 reminders / 100 turns)

Outcomes and what they would mean:

    Both rescue compliance:    attention-fade is real and easily countered;
                               recommend reminders as standard practice.
    Only k=10 rescues:         attention-fade is real but fast; agent
                               developers need a tighter reminder cadence
                               than they might guess.
    Neither rescues:           long-N drift is NOT attention fade. Move to
                               candidate B (reminder inside tool_result
                               content) or C (mid-trajectory primer
                               reinjection), since this falsifies the
                               most likely class of intervention.

N=100, 10 reps each at K=10 and K=25. ~$6 total. If a condition rescues
N=100, follow up with the same condition at N=200 in a separate run.

The script reaches into adapter internals for the same reasons described
in `_primer_runner.py` — the intervention is at the message-list level
and there is no public Task hook for periodic user-content injection.
The agent loop is inlined rather than extending `_primer_runner` because
the periodic-injection logic is the whole point of this experiment and
belongs alongside its own description.
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
K_VALUES = [10, 25]

REMINDER_TEXT = (
    "Reminder: continue to start each text response with `STATUS: <N>` "
    "where <N> increments by 1 each response."
)


def run_with_periodic_reminder(
    task: Any,
    *,
    k: int,
    model: str,
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
) -> Trajectory:
    """Run the task, appending REMINDER_TEXT to every K-th tool-result user message.

    The reminder is added as a second content block alongside the
    tool_result block, so the agent sees `[tool_result, text]` rather
    than just `[tool_result]` on iterations where (i+1) % k == 0. For
    K=10 this fires after iterations 9, 19, 29, ... — priming assistant
    turns 11, 21, 31, ...

    Trajectory.metadata records the K value and the count of reminders
    actually delivered (which can be less than max_iterations//K if the
    agent finishes early or hits an empty-tool-uses stop).
    """
    trajectory = Trajectory(task_id=task.id, model=model)
    for key, value in task.trajectory_metadata.items():
        trajectory.metadata[key] = value
    trajectory.add_turn("system", content=task.system_prompt)
    trajectory.add_turn("user", content=task.initial_user_message)

    anthropic_tools = _to_anthropic_tools(task.tool_specs, cache=True)
    system_param: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": task.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task.initial_user_message},
    ]
    usage_totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "n_requests": 0,
    }
    reminders_delivered = 0

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
        if (i + 1) % k == 0:
            user_content.append({"type": "text", "text": REMINDER_TEXT})
            reminders_delivered += 1
        messages.append({"role": "user", "content": user_content})

        if task.is_done():
            break

    trajectory.metadata["usage_totals"] = usage_totals
    trajectory.metadata["reminder_k"] = k
    trajectory.metadata["reminder_text"] = REMINDER_TEXT
    trajectory.metadata["reminders_delivered"] = reminders_delivered
    return trajectory


def run_k(k: int, client: anthropic.Anthropic) -> None:
    out_path = Path(f"results/mechanism_reminder/{MODEL}_N{N}_k{k}.jsonl")
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
        print(f"[k={k}] all reps done", file=sys.stderr)
        return

    print(
        f"[k={k}] running {len(plan)} reps on {MODEL} N={N}",
        file=sys.stderr,
        flush=True,
    )
    for rep in plan:
        print(f"  + reminder k={k} N={N} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(N, seed=rep, n_plants=1, discovery=False)
        trajectory = run_with_periodic_reminder(
            task, k=k, model=MODEL, client=client, max_iterations=500
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": N,
            "rep": rep,
            "k": k,
            "model": MODEL,
            "condition": f"periodic_reminder_k{k}",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    client = anthropic.Anthropic(max_retries=20, timeout=300.0)
    for k in K_VALUES:
        run_k(k, client)
    print("all k values done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
