"""Does the cold-start cure generalise from format rules to content rules?

The mechanism investigation found that on Sonnet 4.6's STATUS counter rule,
two trivial off-topic user/assistant exchanges drive N=10 compliance from
0.58 to 1.00 — and the format does not need to appear in the primer
responses for the intervention to work. That's *one rule on one model*.
This script tests whether the cure generalises to a structurally different
rule: a content keyword that must appear somewhere in every response.

Rule: "include the word 'investigating' in every response."

Structurally different from the STATUS counter on three axes:

- *Type*: content vs format
- *Position*: anywhere vs prefix
- *State*: stateless vs counter (no integer to track)

If the four primer conditions produce the same dose-response on this rule
(baseline bimodal, zero tri-modal, minimal mostly-bimodal, full perfect),
the cold-start dynamic is a property of how Sonnet engages with system-
prompt rules in general, not a quirk of the counter rule. If they don't,
the cure is specific to format/structural rules and the framework needs
to be narrowed.

10 reps at N=10 across 4 conditions = 40 trajectories. ~$1.50.
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

MODEL = "claude-sonnet-4-6"
KEYWORD = "investigating"

CONTENT_RULE_PROMPT_HEADER = (
    "You are a research assistant.\n\n"
    "OUTPUT FORMAT (must be followed every single turn): the word "
    f"'{KEYWORD}' must appear somewhere in every text response you "
    "produce. The word can appear anywhere in the response — beginning, "
    "middle, or end. Use it naturally; you may add it to whatever you "
    "would have said anyway.\n\n"
    "Example of a correctly formatted assistant response:\n"
    f"  Investigating topic_3 now. Looking up the value.\n\n"
)


def override_task_with_content_rule(task: Any) -> None:
    """Mutate the find_and_synthesise task to use the content keyword rule."""
    # Rebuild the system prompt: replace the STATUS rule with the content rule.
    base_prompt = task.system_prompt
    task_section_start = base_prompt.find("TASK:")
    if task_section_start == -1:
        raise RuntimeError("Could not find 'TASK:' header in task system_prompt")
    task_section = base_prompt[task_section_start:]
    task.system_prompt = CONTENT_RULE_PROMPT_HEADER + task_section

    # Replace the trajectory_metadata so instruction_decay uses our rule.
    task.trajectory_metadata["instruction_decay"] = {
        "rule_id": "contains_keyword",
        "params": {"keyword": KEYWORD, "case_sensitive": False},
    }


CONDITIONS: dict[str, list[dict[str, Any]]] = {
    "baseline": [],
    "zero": [
        {"role": "user", "content": "Hi."},
    ],
    "minimal": [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "ok."},
    ],
    "full": [
        {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 is 4."},
        {"role": "user", "content": "Another warm-up: what is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
}


def run_with_primer(
    task: Any,
    *,
    primer: list[dict[str, Any]],
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
        response = cast(
            "Message",
            client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system_param,
                messages=messages,
                tools=anthropic_tools,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            ),
        )
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


def run_condition(
    cond_name: str, primer: list[dict[str, Any]], client: anthropic.Anthropic
) -> None:
    out_path = Path(f"results/mechanism_content/{cond_name}_{MODEL}.jsonl")
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
        f"[{cond_name}] running {len(plan)} reps on {MODEL} (content-keyword rule)",
        file=sys.stderr,
        flush=True,
    )
    for rep in plan:
        print(f"  + {cond_name} N=10 rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(10, seed=rep, n_plants=1, discovery=False)
        override_task_with_content_rule(task)
        trajectory = run_with_primer(task, primer=primer, client=client)
        scores: Mapping[str, Score] = {
            name: probe(trajectory) for name, probe in PROBES.items()
        }
        record: dict[str, Any] = {
            "n": 10,
            "rep": rep,
            "model": MODEL,
            "rule": "contains_keyword/investigating",
            "condition": cond_name,
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)
    for cond_name, primer in CONDITIONS.items():
        run_condition(cond_name, primer, client)
    print("all conditions done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
