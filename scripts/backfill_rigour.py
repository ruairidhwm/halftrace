"""Backfill 5 more reps on the cells that need them for rigour.

After review I'd want to tighten three things:

1. Sonnet N=200 missing 2 reps (currently 8/10, the rest of Sonnet is 10/10).
2. Opus, Haiku, GPT-4.1, GPT-4o at N=25 — bring 5→10 reps (tightens the
   "perfect" CI from ruling out >45% failure to ruling out >25%).
3. Same four at N=100 — same reason.

These are the cells where the marginal rigour-per-dollar is highest.
Skips cells already at the target rep count.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anthropic
import openai

from halftrace.adapters.anthropic_adapter import run_anthropic_task
from halftrace.adapters.openai_adapter import run_openai_task
from halftrace.probes import (
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.probes.base import Score
from halftrace.tasks import find_and_synthesise

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}

# (model, n_value, atlas_subdir, target_reps)
PLAN: list[tuple[str, int, str, int]] = [
    # Sonnet N=200: finish to 10 (currently 8)
    ("claude-sonnet-4-6", 200, "atlas_deep", 10),
    # Bring the "perfect" models to 10 reps at N=25 and N=100
    ("claude-opus-4-7", 25, "atlas", 10),
    ("claude-opus-4-7", 100, "atlas_deep", 10),
    ("claude-haiku-4-5", 25, "atlas", 10),
    ("claude-haiku-4-5", 100, "atlas_deep", 10),
    ("gpt-4.1", 25, "atlas", 10),
    ("gpt-4.1", 100, "atlas_deep", 10),
    ("gpt-4o", 25, "atlas", 10),
    ("gpt-4o", 100, "atlas_deep", 10),
]


def _load_existing(path: Path, n: int) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["rep"])
        for line in path.read_text().splitlines()
        if line.strip() and int(json.loads(line)["n"]) == n
    }


def _run_one(client: object, model: str, n: int, rep: int):
    task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
    if model.startswith("claude-"):
        trajectory = run_anthropic_task(
            task, model=model, max_tokens=4096, max_iterations=500,
            client=client,  # type: ignore[arg-type]
            disable_parallel_tool_use=True,
        )
    else:
        trajectory = run_openai_task(
            task, model=model, max_tokens=4096, max_iterations=500,
            client=client,  # type: ignore[arg-type]
            disable_parallel_tool_use=True,
        )
    scores: Mapping[str, Score] = {
        name: probe(trajectory) for name, probe in PROBES.items()
    }
    return trajectory, scores


def main() -> int:
    anthropic_client: anthropic.Anthropic | None = None
    openai_client: openai.OpenAI | None = None

    for model, n, subdir, target_reps in PLAN:
        out_path = Path(f"results/{subdir}/{model}.jsonl")
        existing = _load_existing(out_path, n)
        missing = [rep for rep in range(target_reps) if rep not in existing]
        if not missing:
            print(f"  [skip] {model} N={n}: already at {len(existing)} reps", file=sys.stderr)
            continue
        print(
            f"\n=== {model} N={n} ({len(existing)} → {target_reps}) ===",
            file=sys.stderr, flush=True,
        )

        if model.startswith("claude-"):
            if anthropic_client is None:
                anthropic_client = anthropic.Anthropic(max_retries=20, timeout=180.0)
            client: object = anthropic_client
        else:
            if openai_client is None:
                openai_client = openai.OpenAI(max_retries=30, timeout=300.0)
            client = openai_client

        for rep in missing:
            print(f"  + N={n} rep={rep}", file=sys.stderr, flush=True)
            trajectory, scores = _run_one(client, model, n, rep)
            record: dict[str, Any] = {
                "n": n, "rep": rep, "model": model,
                "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
                "trajectory": trajectory.model_dump(mode="json"),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    print("\n=== backfill done ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
