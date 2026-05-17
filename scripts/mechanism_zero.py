"""Zero-content primer: is the assistant turn the load-bearing piece?

The minimal-primer control (scripts/mechanism_minimal.py) showed a clean
dose-response on Sonnet 4.6 N=10 instruction compliance:

    baseline (no primer):                 0.345 mean, 1/5 perfect
    one trivial exchange (Hi. / ok.):     0.836 mean, 8/10 perfect
    two trivial exchanges (no format):    1.000 mean, 10/10 perfect

That establishes the effect is content-independent and quantitative.
Open question: is *any* prior turn enough, or specifically a prior
*assistant* turn? This script isolates that. The primer here is a single
extra user message before the real task's user message — no prior
assistant turn at all.

If compliance stays near baseline, the assistant turn is doing the
work: the agent commits more reliably when it has at least one prior
turn of *its own* in the visible context. If compliance jumps toward
1.0, what matters is just the position of the real task within the
conversation, not who spoke before.

Note: Anthropic's API accepts consecutive same-role messages without
merging them on the client side. The two `user` messages are sent as
distinct turns.
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

# ZERO-CONTENT: one extra user turn before the real task's user message.
# No prior assistant turn anywhere in the visible context.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Hi."},
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
    """Run task with a single extra user message prepended (no assistant turn)."""

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
    out_path = Path("results/mechanism_zero/claude-sonnet-4-6.jsonl")
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
        print("all mechanism zero cells done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} ZERO N=10 sonnet trajectories (extra user msg, no assistant primer)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for rep in plan:
        print(f"  + zero N=10 rep={rep}", file=sys.stderr, flush=True)
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
            "condition": "zero_extra_user_only",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
