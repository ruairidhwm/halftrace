"""Minimal-primer control: how degenerate can the prior turn be?

The mechanism test (scripts/mechanism_test.py) and its no-format control
(scripts/mechanism_control.py) both drove Sonnet 4.6's N=10 instruction
compliance to 10/10 perfect, against a baseline of 1/5 perfect. That
falsifies the strong "agent commits based on seeing the format in prior
turns" hypothesis: two off-topic exchanges (2+2, capital of Paris) with
plain-prose assistant responses already close the gap.

This script tightens the screw. The primer is the smallest possible
prior exchange: one user message, one trivial assistant reply ("ok.").
If 10/10 still holds, the warm-up effect is about presence-of-prior-turn,
not coherence-of-prior-content. If compliance regresses toward the
0.345 baseline, two exchanges of substance matter and the one-turn
primer isn't enough.
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

# MINIMAL: the smallest possible prior exchange. One user turn, one
# trivial assistant reply, no format, no content of substance.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Hi."},
    {"role": "assistant", "content": "ok."},
]


def run_primed_task(
    task: Any,
    *,
    model: str,
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
    disable_parallel_tool_use: bool = True,
) -> Trajectory:
    """Run task with a minimal one-turn primer prepended to messages."""

    trajectory = Trajectory(task_id=task.id, model=model)
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
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": messages,
        }
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
            if disable_parallel_tool_use:
                create_kwargs["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
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


def main() -> int:
    out_path = Path("results/mechanism_minimal/claude-sonnet-4-6.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(int(r["rep"]))

    plan = [rep for rep in range(10) if rep not in done]
    if not plan:
        print("all mechanism minimal cells done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} MINIMAL N=10 sonnet trajectories (one-turn primer)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for rep in plan:
        print(f"  + minimal N=10 rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(10, seed=rep, n_plants=1, discovery=False)
        trajectory = run_primed_task(
            task,
            model="claude-sonnet-4-6",
            client=client,
        )
        scores: Mapping[str, Score] = {
            name: probe(trajectory) for name, probe in PROBES.items()
        }
        record: dict[str, Any] = {
            "n": 10,
            "rep": rep,
            "model": "claude-sonnet-4-6",
            "condition": "minimal_one_turn_primer",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
