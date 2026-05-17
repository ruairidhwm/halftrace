"""Haiku 4.5 replication of the Sonnet primer dose-response.

Sonnet 4.6 N=10 instruction compliance on the STATUS counter rule:

    baseline      0.345  bimodal (1/5 commit, 4/5 abandon)
    zero          0.755  tri-modal (commit / alternate / abandon)
    minimal       0.836  bimodal (8/10 commit, 2/10 abandon)
    full          1.000  always commit

Open question: does this generalize to Haiku, or is it a Sonnet-specific
commit dynamic? Haiku 4.5 was previously tested only on the marker rule
(scores 0.000 everywhere) — never on the counter rule, and never with
any primer condition. This script runs all four Haiku conditions in one
pass for a clean comparison.

Each condition writes 10 reps at N=10 to results/mechanism_<cond>/claude-haiku-4-5.jsonl.
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

MODEL = "claude-haiku-4-5"

CONDITIONS: dict[str, tuple[str, list[dict[str, Any]]]] = {
    # cond name -> (out subdir, primer messages)
    "baseline": ("mechanism_baseline_haiku", []),
    "zero": (
        "mechanism_zero",
        [{"role": "user", "content": "Hi."}],
    ),
    "minimal": (
        "mechanism_minimal",
        [
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "ok."},
        ],
    ),
    "full": (
        "mechanism_control",
        [
            {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 is 4."},
            {"role": "user", "content": "Another warm-up: what is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    ),
}


def run_with_primer(
    task: Any,
    *,
    primer: list[dict[str, Any]],
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
    disable_parallel_tool_use: bool = True,
) -> Trajectory:
    """Run task with the given primer prepended to the message list."""

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
        *primer,
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
    trajectory.metadata["primer_messages"] = primer
    return trajectory


def run_condition(cond_name: str, client: anthropic.Anthropic) -> None:
    subdir, primer = CONDITIONS[cond_name]
    out_path = Path(f"results/{subdir}/{MODEL}.jsonl")
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
        print(f"[{cond_name}] all reps done", file=sys.stderr)
        return

    print(
        f"[{cond_name}] running {len(plan)} reps on {MODEL}",
        file=sys.stderr,
        flush=True,
    )

    for rep in plan:
        print(f"  + {cond_name} N=10 rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(10, seed=rep, n_plants=1, discovery=False)
        trajectory = run_with_primer(task, primer=primer, client=client)
        scores: Mapping[str, Score] = {
            name: probe(trajectory) for name, probe in PROBES.items()
        }
        record: dict[str, Any] = {
            "n": 10,
            "rep": rep,
            "model": MODEL,
            "condition": cond_name,
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)
    for cond_name in CONDITIONS:
        run_condition(cond_name, client)
    print("all conditions done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
