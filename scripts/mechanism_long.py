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
from pathlib import Path
from typing import Any

import anthropic
from _primer_runner import run_with_primer, score_all

from halftrace.tasks import find_and_synthesise

MODEL = "claude-sonnet-4-6"
CONDITION = "long_full_primer"
N_VALUES = [100, 200]
REPS = 10
# Full no-format primer — the strongest condition from mechanism_control.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 is 4."},
    {"role": "user", "content": "Another warm-up: what is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


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

    print(
        f"[N={n}] running {len(plan)} primed reps on {MODEL}",
        file=sys.stderr,
        flush=True,
    )
    for rep in plan:
        print(f"  + long N={n} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
        trajectory = run_with_primer(
            task,
            primer=PRIMER_MESSAGES,
            model=MODEL,
            client=client,
            max_iterations=500,
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": n,
            "rep": rep,
            "model": MODEL,
            "condition": CONDITION,
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
