"""Cross-model atlas pilot.

Sweeps five frontier models on find_and_synthesise at three N values with
five reps each, then emits a per-(model, probe) shape + commit_probability
table. The screenshot artefact for the blog post and the README.

Run with: `uv run python scripts/atlas_pilot.py`

The pilot is sequential across models; each model writes its own JSONL to
results/atlas/<model>.jsonl. A combined summary lands at
results/atlas/summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from halftrace.runner import default_trial, run_pilot

MODELS: list[str] = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5",
    "gpt-4.1",
    "gpt-4o",
]
N_VALUES: list[int] = [5, 10, 25]
REPS: int = 5
SERIAL: bool = True


def main() -> int:
    output_dir = Path("results/atlas")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    total_cost = 0.0

    for model in MODELS:
        print(f"\n=== {model} ===", file=sys.stderr, flush=True)
        out_path = output_dir / f"{model.replace('/', '_')}.jsonl"
        trial = default_trial(
            model=model,
            max_tokens=4096,
            max_iterations=500,
            serial=SERIAL,
            n_plants=1,
            discovery=False,
        )
        result = run_pilot(
            model=model,
            n_values=N_VALUES,
            reps=REPS,
            output_path=out_path,
            trial=trial,
            progress=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
        cost = result.estimated_cost_usd or 0.0
        total_cost += cost
        print(
            f"  trajectories={result.n_trajectories}  "
            f"in={result.total_input_tokens:,}  "
            f"out={result.total_output_tokens:,}  "
            f"cost=${cost:.3f}",
            file=sys.stderr,
        )
        for probe_name, profile in result.profiles.items():
            if profile is None:
                continue
            summary.append(
                {
                    "model": model,
                    "probe": probe_name,
                    "shape": profile.shape,
                    "commit_p": round(profile.commit_probability, 3),
                    "halftrace": profile.halftrace,
                    "n_trajectories": sum(profile.n_observations_by_n.values()),
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Atlas complete ===", file=sys.stderr)
    print(f"Total cost: ${total_cost:.2f}", file=sys.stderr)
    print(f"Summary written to {summary_path}", file=sys.stderr)

    print("\n=== Atlas table ===", file=sys.stderr)
    print(
        f"{'model':<22} {'probe':<24} {'shape':<14} {'commit_p':<10}",
        file=sys.stderr,
    )
    for row in summary:
        line = (
            f"{row['model']!s:<22} {row['probe']!s:<24} "
            f"{row['shape']!s:<14} {row['commit_p']!s:<10}"
        )
        print(line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
