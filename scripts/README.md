# scripts/

Reproduction artefacts for the halftrace pilot phase and follow-up
mechanism investigation. These are **ad-hoc experiment runners**, not
part of the published library — they are not shipped in the wheel and
strict pyright is disabled on this directory (see `pyproject.toml`).

The library (`src/halftrace/`) is the supported public surface. These
scripts deliberately reach into adapter internals (e.g. importing private
`_to_anthropic_tools` / `_parse_response` helpers) to thread custom
primer messages through, which is exactly what a real consumer of the
library should never do.

Each script appends one line of JSON per trajectory to a file under
`results/`. Re-running any script is **resume-aware**: previously-
completed `(N, rep)` cells are skipped. Set `ANTHROPIC_API_KEY` (and for
the gpt-3.5-turbo extension, `OPENAI_API_KEY`) in `.env` at the repo
root before running:

```
set -a && source .env && set +a
uv run python scripts/<script_name>.py
```

## Atlas

The cross-model atlas that powers the headline finding in
[`paper/note.md` §3.2-3.4](../paper/note.md) and the
[blog post](https://ruairidh.dev/blog/modern-claude-commits-or-abandons).

| Script | Produces | Cost | Notes |
| --- | --- | --- | --- |
| [`atlas_rigour.py`](atlas_rigour.py) | `results/atlas/<model>.jsonl`, `results/atlas_deep/<model>.jsonl`, `results/atlas_intermediate/claude-sonnet-4-6.jsonl` | ~$40 | Canonical atlas entry point. Five frontier models × N∈{5,10,25,35,50,70,100,200} × 10 reps with resume support. Brings every cell to 10 reps and adds Sonnet intermediate N values. |
| [`atlas_gpt35.py`](atlas_gpt35.py) | `results/atlas_gpt35/gpt-3.5-turbo.jsonl` | ~$1 | The cross-model extension to gpt-3.5-turbo introduced in v0.1.1 / [`paper/note.md` §3.8](../paper/note.md). Hits OpenAI quota at N=100; the first four N values complete cleanly. |
| [`backfill_rigour.py`](backfill_rigour.py) | (multiple `results/atlas*/`) | varies | Utility to top up any cells that drift below 10 reps after an interrupted run. Use after a credit-exhaustion or rate-limit failure. |

Earlier atlas iteration scripts (debugging artefacts, finishers for
individual model cells, etc.) are in [`archive/`](archive/) for
historical reference.

## Mechanism investigation

The four-condition (× two rules) primer investigation that produced
[`paper/note.md` §3.5-3.7](../paper/note.md). Each script writes 10 reps
at N=10 on Sonnet 4.6 unless otherwise noted; together they cost ~$5.

All eight share [`_primer_runner.py`](_primer_runner.py) for the
prepend-primer-then-run-the-task loop. Each script is just a configuration
of primer messages + output path + optional task override.

| Script | Primer | Rule | Result |
| --- | --- | --- | --- |
| [`mechanism_test.py`](mechanism_test.py) | two exchanges with `STATUS: 1` / `STATUS: 2` on assistant turns | counter (STATUS) | 10/10 perfect (mean 1.00) |
| [`mechanism_control.py`](mechanism_control.py) | two exchanges, no STATUS on assistant turns | counter | 10/10 perfect (mean 1.00) — falsifies the "format-in-priors" hypothesis |
| [`mechanism_minimal.py`](mechanism_minimal.py) | one trivial exchange (`Hi.`/`ok.`) | counter | 8/10 perfect (mean 0.84) — dose-response: one assistant turn is mostly enough |
| [`mechanism_zero.py`](mechanism_zero.py) | one extra user message, no assistant turn | counter | tri-modal (mean 0.76) — the assistant turn is the load-bearing piece |
| [`mechanism_haiku.py`](mechanism_haiku.py) | all four conditions | counter | 40/40 perfect on Haiku 4.5 — no baseline failure to cure |
| [`mechanism_content_rule.py`](mechanism_content_rule.py) | all four conditions | content keyword `investigating` | 10/10 perfect at baseline — no failure to cure |
| [`mechanism_marker_rule.py`](mechanism_marker_rule.py) | all four conditions | marker `[[END]]` | non-monotone dose-response; zero condition worsens compliance |
| [`mechanism_long.py`](mechanism_long.py) | full primer at N=100 and N=200 | counter | cure fails at long N (mean 0.47 at N=100, mean 0.01 at N=200) |

`_primer_runner.py` exports `run_with_primer(task, *, primer, model, client)`
and `score_all(trajectory)`. To add a ninth mechanism condition, write a
~50-line script in the shape of `mechanism_test.py`.

## Adversarial task

| Script | Produces | Notes |
| --- | --- | --- |
| [`find_max_pilot.py`](find_max_pilot.py) | `results/find_max/n=<N>/<model>.jsonl` | Pilot the `find_max` task (introduced in v0.1.1) on one or more models. Exercises the `premature_termination` probe. CLI: `--models claude-sonnet-4-6 --n 10 --reps 10`. Sonnet 4.6 at N=10 and N=30 scores 1.0 universally — the probe is mechanically valid but the failure mode is not elicited on this model on this task. |

## Cost ledger (cumulative across all scripts)

See [RESULTS.md § Cost ledger](../RESULTS.md#cost-ledger) for the
authoritative table. Approximate totals as of v0.1.1:

- Atlas (`atlas_rigour.py` + `atlas_gpt35.py`): ~$41
- Mechanism investigation (eight `mechanism_*.py`): ~$5
- find_max pilot at N=10 and N=30: ~$0.85
- **Total ≈ $47** across ~370 trajectories.
