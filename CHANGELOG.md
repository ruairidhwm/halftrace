# Changelog

All notable changes to halftrace are recorded here. This project follows
[semantic versioning](https://semver.org/).

## v0.1.1 — 2026-05-17

### Added

- **`premature_termination` probe** — completes the five-probe set from the
  pre-registration. Scores whether the agent finished before completing the
  required work, based on a `{n_required, n_completed}` annotation that the
  task records when the terminating tool is called. The probe checks
  trajectory-level metadata first, then falls back to scanning tool-result
  turn metadata so tasks that record completion data dynamically can score
  cleanly.
- **`FindMax` / `find_max` task** — adversarial companion task that admits
  the `premature_termination` failure mode. Agent gets N topics with hidden
  integer values, is asked to find the maximum, and is given explicit
  licence to be efficient. Agents that take the licence submit before
  exhausting lookups; the probe scores the resulting `n_completed /
  n_required`. Optional `instruction_decay_rule` flag layers the STATUS
  counter rule onto the same task.
- **`from_langsmith_run` ingest** — converts a LangSmith trace run dict
  into a halftrace `Trajectory`. Walks the run tree, locates the final
  LLM run, extracts `inputs.messages` (OpenAI format), synthesises a
  final assistant turn from `outputs.generations[*].message`, preserves
  `tool_calls`. Handles both `child_runs` and `children` keys; extracts
  model name from `extra.invocation_params` or `serialized.kwargs`.
- **`contains_keyword` matcher** on the `instruction_decay` probe — case-
  insensitive substring presence check with a `case_sensitive=True`
  override. For testing content rules ("include the word X in every
  response") alongside the existing `end_with_marker` and
  `starts_with_pattern` matchers.
- **gpt-3.5-turbo** added to the pricing table; the OpenAI adapter
  already dispatched on the `gpt-*` prefix so the model worked, only the
  cost estimate was missing.

### Changed

- **`diagnose()` for the `bimodal` shape** — the suggestion catalog is now
  grounded in the post-v0.1.0 mechanism investigation. The warm-up cure is
  framed as short-N-specific; new advice is included for long-trajectory
  abandonment (periodic rule restatement, system-prompt reinjection) and
  for parallel interventions when the developer only controls the system
  prompt.

### Findings released alongside v0.1.1

This release bundles four new experiments that together narrow the cold-
start mechanism finding from v0.1.0:

1. **Cure at long N** — 10 reps Sonnet at N=100 with the full two-exchange
   primer scored mean 0.470 (vs atlas baseline 0.508). 10 reps at N=200
   scored mean 0.011 with 10/10 abandons (vs 8/10 baseline). The cure is
   N=10-specific; long-N drift is a separate failure mode.
2. **Marker rule replication** — same four primer conditions on the
   front-loaded `[[END]]` rule produced a non-monotone dose-response:
   baseline 0.79, zero 0.58, minimal 0.79, full 0.97. The zero-content
   primer that helped on the counter rule worsens compliance on the
   marker rule. Only the full primer is robustly helpful across rules.
3. **gpt-3.5-turbo atlas** — 40 trajectories at N≤50 confirm the "no
   gradient decay" finding extends below the frontier-model class. A
   categorical "fail first turn then comply" signature dominates instead.
4. **find_max pilot on Sonnet** — at N=10 and N=30, Sonnet 4.6 completed
   every lookup before submitting; `premature_termination` scored 1.000
   in every rep. Probe is mechanically valid; failure mode is not
   elicited on this model on this task.

Full per-cell data in [`RESULTS.md`](RESULTS.md) and discussion in
[`paper/note.md`](paper/note.md).

## v0.1.0 — 2026-05-15

Initial public release. Four probes (`state_amnesia`, `instruction_decay`,
`tool_repetition`, `narration_substitution`), one task
(`find_and_synthesise`), Anthropic + OpenAI adapters, `halftrace`
CLI with `pilot`, `analyse`, and `compare` subcommands, shape classifier
(`analyse_compliance`), and `diagnose()` actionable suggestions.
