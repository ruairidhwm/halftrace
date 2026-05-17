"""gpt-3.5-turbo atlas: does gradient decay show up on a weaker model?

The atlas across five frontier models (Opus 4.7, Sonnet 4.6, Haiku 4.5,
GPT-4.1, GPT-4o) produced *zero* gradient decay on any probe at any N. The
strongest reading of that result is that smooth context-rot is a smaller-
or-older-model phenomenon that frontier models have already solved.

gpt-3.5-turbo is the cheapest test of that reading: same OpenAI adapter,
no new API setup, meaningfully weaker than 4o/4.1. If gradient decay
appears here, the halftrace scalar gets a non-vacuous use case and the
"no decay on modern Claude" claim is correctly framed as model-class
specific. If it doesn't, the negative finding extends beyond the frontier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from halftrace.runner import default_trial, run_pilot

MODEL = "gpt-3.5-turbo"
N_VALUES: list[int] = [5, 10, 25, 50, 100]
REPS: int = 10
SERIAL: bool = True


def main() -> int:
    output_dir = Path("results/atlas_gpt35")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{MODEL}.jsonl"

    print(f"=== {MODEL} ===", file=sys.stderr, flush=True)
    print(
        f"N_VALUES={N_VALUES}  reps={REPS}  serial={SERIAL}",
        file=sys.stderr,
        flush=True,
    )

    trial = default_trial(
        model=MODEL,
        max_tokens=4096,
        max_iterations=500,
        serial=SERIAL,
        n_plants=1,
        discovery=False,
    )
    result = run_pilot(
        model=MODEL,
        n_values=N_VALUES,
        reps=REPS,
        output_path=out_path,
        trial=trial,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )

    cost = result.estimated_cost_usd or 0.0
    print(
        f"\ntrajectories={result.n_trajectories}  "
        f"in={result.total_input_tokens:,}  "
        f"out={result.total_output_tokens:,}  "
        f"cost=${cost:.3f}",
        file=sys.stderr,
    )

    summary: list[dict[str, object]] = []
    for probe_name, profile in result.profiles.items():
        if profile is None:
            continue
        summary.append(
            {
                "model": MODEL,
                "probe": probe_name,
                "shape": profile.shape,
                "commit_p": round(profile.commit_probability, 3),
                "halftrace": profile.halftrace,
                "means_by_n": {n: round(m, 3) for n, m in profile.means_by_n.items()},
                "variances_by_n": {
                    n: round(v, 4) for n, v in profile.variances_by_n.items()
                },
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {summary_path}", file=sys.stderr)

    for row in summary:
        print(
            f"  {row['probe']:<24} shape={row['shape']:<13} "
            f"commit_p={row['commit_p']}  means={row['means_by_n']}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
