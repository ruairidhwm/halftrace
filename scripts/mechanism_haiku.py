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

Each condition writes 10 reps at N=10 to
results/mechanism_<cond>/claude-haiku-4-5.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anthropic
from _primer_runner import run_with_primer, score_all

from halftrace.tasks import find_and_synthesise

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
            {
                "role": "user",
                "content": "Quick warm-up before the real task: what is 2 + 2?",
            },
            {"role": "assistant", "content": "2 + 2 is 4."},
            {
                "role": "user",
                "content": "Another warm-up: what is the capital of France?",
            },
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    ),
}


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
        trajectory = run_with_primer(
            task, primer=primer, model=MODEL, client=client
        )
        scores = score_all(trajectory)
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
