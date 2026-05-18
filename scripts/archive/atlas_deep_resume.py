"""Resume the deep atlas pilot, skipping (model, N, rep) cells already on disk.

Reads results/atlas_deep/<model>.jsonl for each model, figures out which
trajectories are already complete, and runs only the missing ones. Use this
after a partial failure (rate limit, credit exhaustion, etc.) to avoid
re-spending on data we already have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
from halftrace.tasks import find_and_synthesise

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}

# Per-model plan: which (N, rep) cells should the model have, ordered.
# gpt-4o skips N=200 due to OpenAI's 30K TPM ceiling on the standard tier.
PLAN: dict[str, list[tuple[int, int]]] = {
    "claude-sonnet-4-6": [(n, r) for n in [50, 100, 200] for r in range(5)],
    "claude-opus-4-7": [(n, r) for n in [50, 100, 200] for r in range(5)],
    "claude-haiku-4-5": [(n, r) for n in [50, 100, 200] for r in range(5)],
    "gpt-4.1": [(n, r) for n in [50, 100, 200] for r in range(5)],
    "gpt-4o": [(n, r) for n in [50, 100] for r in range(5)],
}


def _load_existing(path: Path) -> set[tuple[int, int]]:
    """Return the (N, rep) cells already present in the file."""
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((int(r["n"]), int(r["rep"])))
    return done


def _client_for(model: str) -> object:
    if model.startswith("claude-"):
        return anthropic.Anthropic(max_retries=20, timeout=180.0)
    return openai.OpenAI(max_retries=30, timeout=300.0)


def _run_one(model: str, client: object, n: int, rep: int):
    task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
    if model.startswith("claude-"):
        trajectory = run_anthropic_task(
            task,
            model=model,
            max_tokens=4096,
            max_iterations=500,
            client=client,  # type: ignore[arg-type]
            disable_parallel_tool_use=True,
        )
    else:
        trajectory = run_openai_task(
            task,
            model=model,
            max_tokens=4096,
            max_iterations=500,
            client=client,  # type: ignore[arg-type]
            disable_parallel_tool_use=True,
        )
    scores = {name: probe(trajectory) for name, probe in PROBES.items()}
    return trajectory, scores


def main() -> int:
    output_dir = Path("results/atlas_deep")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model, plan in PLAN.items():
        out_path = output_dir / f"{model}.jsonl"
        done = _load_existing(out_path)
        missing = [cell for cell in plan if cell not in done]

        if not missing:
            print(
                f"\n=== {model} (all {len(plan)} cells already done) ===",
                file=sys.stderr,
            )
            continue

        print(
            f"\n=== {model} (have {len(done)}/{len(plan)}; running {len(missing)} missing) ===",
            file=sys.stderr,
            flush=True,
        )
        client = _client_for(model)

        for n, rep in missing:
            print(f"[+] N={n} rep={rep}", file=sys.stderr, flush=True)
            trajectory, scores = _run_one(model, client, n, rep)
            record = {
                "n": n,
                "rep": rep,
                "model": model,
                "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
                "trajectory": trajectory.model_dump(mode="json"),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    print("\n=== Resume run complete ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
