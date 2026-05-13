# Pre-registered hypotheses: the context rot study

## Experimental setup

- **Task:** `find_and_synthesise` — a synthetic agentic task scaling by N (the number of tool calls required to complete the task).
- **N values:** {5, 10, 25, 50, 100, 200}
- **Models:** claude-sonnet-4-6, claude-opus-4-7, gpt-4o, gpt-4.1
- **Probes:** state_amnesia, instruction_decay, tool_repetition, premature_termination, narration_substitution
- **Repetitions:** 5 per (model, N, probe) cell
- **Judge:** claude-haiku-4-5, temperature 0
- **Total trajectories:** 6 × 5 × 5 = 120

A halftrace is the trajectory length at which a probe score crosses 0.5, fitted by linear interpolation across N values with 95% confidence intervals from bootstrap resampling over the 5 repetitions.

## Primary hypothesis

**H1: The four failure modes have meaningfully different halftraces.**

Specifically, I predict the ordering from shortest halftrace (degrades fastest) to longest (most robust), aggregated across models:

1. `instruction_decay` — halftrace under N=40 on most models
2. `narration_substitution` — halftrace between N=40 and N=80
3. `tool_repetition` — halftrace between N=60 and N=120
4. `state_amnesia` — halftrace above N=120 on most models
5. `premature_termination` — does not cross 0.5 in the tested range on any model

**H1 is falsified if:**
- `state_amnesia` halftrace is shorter than `instruction_decay` halftrace on the majority of models
- Any two probes have halftraces within 10% of each other on all models (i.e. the four curves collapse to fewer distinct curves)
- The ordering above holds on fewer than three of the five models

## Secondary hypotheses

**H2: Claude Opus 4.7 will be the most robust model overall.**
Specifically, Opus will have the longest halftrace on at least three of the five probes.

**H3: At least one model will show non-monotonic behaviour on at least one probe.**
That is, a probe score that goes down then back up as N increases, probably from the agent skipping work as the context fills. If this happens, it suggests "give up" as a coping strategy rather than continued degradation. It's also cheaper.

**H4: Cross-model variance is larger for `instruction_decay` than for `state_amnesia`.**
Models will differ more in how well they follow rules over time than in how well they remember facts. This would suggest instruction-following is more a property of post-training than of base capability.

**H5: GPT-4.1 will have a shorter `narration_substitution` halftrace than Claude Sonnet 4.6.**
This builds on the narration-substitution failure mode first documented in [Compressing Prompts with an Autoresearch Loop](https://www.ruairidh.dev/blog/compressing-prompts-with-an-autoresearch-loop).

## Known limitations and confounds

1. **Judge model context limits.** The judge (Claude Haiku) has a finite context window; at high N the judge may be evaluating trajectories that approach its own limits. This could artificially depress probe scores at high N independent of the agent's behaviour.
2. **The `find_and_synthesise` task is one task.** Findings may not generalise to other agentic task shapes. The library is designed to make replication on other tasks easy.
3. **Five repetitions is small.** Confidence intervals will be wide. We need to be careful not to over-claim point estimates; emphasis should be on the ordering and the qualitative shape of the curves.
4. **Cross-provider determinism is not guaranteed.** Even with temperature 0 and fixed seeds, OpenAI and Anthropic do not promise bit-exact reproducibility. I'm fairly sure that results from a re-run will differ slightl.
5. **The `narration_substitution` probe is novel.** Its definition was introduced in the compression study but has not been independently validated. Sensitivity analysis on its threshold is required.
6. **No baseline at N=0.** All measurements are relative to N=5 performance, not to a hypothetical zero-context baseline. This understates absolute degradation.

## What would change my mind

Specific things that, if observed, would make me revise my conclusions before publishing:

- All four probe scores correlating above 0.9 across N (suggests one underlying capability is degrading, not four)
- Any model achieving a halftrace > 200 on all four probes (suggests the test isn't hard enough)
- Reps 1–5 within a single cell varying by more than 0.3 on probe score (suggests the test is too noisy to draw conclusions)

## Stopping conditions

The experiment ends when:

- All 150 trajectories are collected, OR
- API spend reaches £200, OR
- A single model's failure rate (HTTP errors, refused completions) exceeds 20% — in which case results for that model are reported with that caveat
