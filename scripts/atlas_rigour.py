"""Tighten the atlas: 10 reps per cell + intermediate N values for Sonnet.

Two goals:

A. Bump reps per cell from 5 to 10 across the whole atlas (shallow N=5/10/25
   in results/atlas/<model>.jsonl plus deep N=50/100/200 in
   results/atlas_deep/<model>.jsonl). With 10 reps a flat-1.0 cell rules out
   a 10% latent failure rate at p<0.05 (one-sided), which we couldn't claim
   with 5 reps.

B. Add intermediate N values for Sonnet — the only model showing decay —
   to characterise the transition between N=25 (perfect) and N=100
   (~10% commit). Adds N=35 and N=70, 10 reps each, writes to
   results/atlas_intermediate/claude-sonnet-4-6.jsonl.

Resume-aware: reads existing data, runs only missing (model, N, rep) cells.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
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
from halftrace.trajectory import Trajectory

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}

ALL_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5",
    "gpt-4.1",
    "gpt-4o",
]

SHALLOW_N = [5, 10, 25]
DEEP_N = [50, 100, 200]
SONNET_INTERMEDIATE_N = [35, 70]
TARGET_REPS = 10


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


def _load_existing(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((int(r["n"]), int(r["rep"])))
    return done


def _append_record(
    path: Path,
    n: int,
    rep: int,
    model: str,
    trajectory: Trajectory,
    scores: Mapping[str, Score],
) -> None:
    record: dict[str, Any] = {
        "n": n,
        "rep": rep,
        "model": model,
        "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
        "trajectory": trajectory.model_dump(mode="json"),
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _run_cells(
    model: str,
    out_path: Path,
    n_values: Iterable[int],
    target_reps: int,
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(out_path)
    plan: list[tuple[int, int]] = [
        (n, rep)
        for n in n_values
        for rep in range(target_reps)
        if (n, rep) not in done
    ]
    if not plan:
        print(f"  [{label}] all cells already complete", file=sys.stderr)
        return
    print(
        f"  [{label}] running {len(plan)} missing cells "
        f"(have {len(done)}, target {target_reps * len(list(n_values))})",
        file=sys.stderr,
        flush=True,
    )
    client = _client_for(model)
    for n, rep in plan:
        print(f"    + N={n} rep={rep}", file=sys.stderr, flush=True)
        trajectory, scores = _run_one(model, client, n, rep)
        _append_record(out_path, n, rep, model, trajectory, scores)


def main() -> int:
    # A: bump reps to 10 on the shallow and deep atlases for every model.
    for model in ALL_MODELS:
        print(f"\n=== {model} ===", file=sys.stderr, flush=True)
        # Shallow
        _run_cells(
            model,
            Path("results/atlas") / f"{model}.jsonl",
            SHALLOW_N,
            TARGET_REPS,
            "shallow",
        )
        # Deep — gpt-4o and gpt-4.1 skip N=200 (gpt-4o hits 30K TPM; gpt-4.1
        # standard-tier reliably hangs in the retry loop at N=200).
        deep_n = [
            n
            for n in DEEP_N
            if not (model in ("gpt-4o", "gpt-4.1") and n == 200)
        ]
        _run_cells(
            model,
            Path("results/atlas_deep") / f"{model}.jsonl",
            deep_n,
            TARGET_REPS,
            "deep",
        )

    # B: intermediate N values for Sonnet only.
    print("\n=== claude-sonnet-4-6 (intermediate N) ===", file=sys.stderr, flush=True)
    _run_cells(
        "claude-sonnet-4-6",
        Path("results/atlas_intermediate") / "claude-sonnet-4-6.jsonl",
        SONNET_INTERMEDIATE_N,
        TARGET_REPS,
        "intermediate",
    )

    print("\n=== Rigour pass complete ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
