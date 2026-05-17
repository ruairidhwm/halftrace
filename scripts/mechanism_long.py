"""Does the cold-start cure also rescue the long-context drift regime?

Sonnet 4.6 instruction compliance on the STATUS counter rule:

    Baseline:  N=10   0.58  (4/10 commit)
               N=100  0.51  (~5/10 commit)
               N=200  0.16  (~1/10 commit)
    Full primer (2 trivial off-topic exchanges, no format in primer responses):
               N=10   1.00  (10/10 commit)

Open question: is the late-trajectory drift (N=100, N=200) the same mechanism
as the early-uncertainty dip (N=10), or a different one? If the same primer
that closed the N=10 dip also rescues N=100 and N=200, the cold-start dynamic
explains the entire Sonnet decay curve and the practical fix is universal.
If it doesn't, late-trajectory abandonment is a different failure mode
(probably system-prompt attention fade) and needs its own intervention.

10 reps at each of N=100 and N=200, Sonnet 4.6, full primer condition.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import anthropic
from anthropic.types import Message, Usage

from halftrace.adapters.anthropic_adapter import (
    _accumulate_usage,
    _parse_response,
    _to_anthropic_tools,
    _to_param_blocks,
)
from halftrace.probes import (
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.probes.base import Score
from halftrace.tasks import find_and_synthesise
from halftrace.trajectory import ToolResult, Trajectory

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}

# Same primer as mechanism_control.py (the strongest non-format condition):
# two off-topic user/assistant exchanges with no STATUS prefix on the
# assistant primer responses.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 is 4."},
    {"role": "user", "content": "Another warm-up: what is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

N_VALUES = [100, 200]
REPS = 10
MODEL = "claude-sonnet-4-6"


def run_primed_task(
    task: Any,
    *,
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
) -> Trajectory:
    trajectory = Trajectory(task_id=task.id, model=MODEL)
    for k, v in task.trajectory_metadata.items():
        trajectory.metadata[k] = v
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
        *PRIMER_MESSAGES,
        {"role": "user", "content": task.initial_user_message},
    ]
    usage_totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "n_requests": 0,
    }

    for _ in range(max_iterations):
        create_kwargs: dict[str, Any] = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": messages,
            "tools": anthropic_tools,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
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
        messages.append({"role": "user", "content": tool_result_blocks})
        if task.is_done():
            break

    trajectory.metadata["usage_totals"] = usage_totals
    trajectory.metadata["primer_messages"] = PRIMER_MESSAGES
    return trajectory


def run_n(n: int, client: anthropic.Anthropic) -> None:
    out_path = Path(f"results/mechanism_long/{MODEL}_N{n}.jsonl")
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
        print(f"[N={n}] all reps done", file=sys.stderr)
        return

    print(f"[N={n}] running {len(plan)} primed reps on {MODEL}", file=sys.stderr, flush=True)
    for rep in plan:
        print(f"  + long N={n} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
        trajectory = run_primed_task(task, client=client)
        scores: Mapping[str, Score] = {
            name: probe(trajectory) for name, probe in PROBES.items()
        }
        record: dict[str, Any] = {
            "n": n,
            "rep": rep,
            "model": MODEL,
            "condition": "long_full_primer",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    client = anthropic.Anthropic(max_retries=20, timeout=300.0)
    for n in N_VALUES:
        run_n(n, client)
    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
