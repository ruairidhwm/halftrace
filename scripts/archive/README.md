# scripts/archive/

Iteration history of the cross-model atlas. None of these scripts are
the canonical entry point — they are debugging artefacts and incremental
runners kept here for reproducibility of the data already in
`results/atlas/`, `results/atlas_deep/`, and `results/atlas_intermediate/`.

If you want to re-run the atlas from scratch, use
[`../atlas_rigour.py`](../atlas_rigour.py) instead. It is resume-aware
and covers everything these earlier scripts collectively produced.

| Script | What it did | Superseded by |
| --- | --- | --- |
| `atlas_pilot.py` | First 5-model sweep at N∈{5,10,25}, 5 reps. The shallow atlas. | `atlas_rigour.py` (bumps to 10 reps) |
| `atlas_finish.py` | Material follow-ups to the shallow atlas. | `atlas_rigour.py` |
| `atlas_deep.py` | Added N∈{50,100,200} to the atlas. The deep atlas. | `atlas_rigour.py` |
| `atlas_deep_gpt.py` | Finished the GPT-4.1/4o half of the deep atlas after Anthropic credit exhaustion mid-run. | `atlas_rigour.py` (resume-aware) |
| `atlas_deep_resume.py` | Generic resume helper for `atlas_deep.py`. | `atlas_rigour.py` |
| `atlas_finish_gpt4o.py` | Filled the GPT-4o cells after OpenAI's 30K-TPM rate limit hit mid-N=25. | `atlas_rigour.py` |
| `atlas_gpt4o_deep.py` | Brought GPT-4o deep cells to parity with the other four models. | `atlas_rigour.py` |
| `sonnet_intermediate.py` | Added N∈{35,70} for Sonnet to characterise the transition between perfect-at-N=25 and ~10%-commit-at-N=100. | `atlas_rigour.py` (folds in intermediate N) |

Why keep them rather than delete? The atlas published in
[`paper/note.md`](../../paper/note.md) was produced by a sequence of
these runs, not by a single `atlas_rigour.py` invocation. A reader
wanting to understand the actual experimental history — which credit
limits hit when, which model required a separate finisher, etc. — gets
that from the docstrings here. Git history would also preserve them,
but moving them out of `scripts/` keeps the active directory readable.
