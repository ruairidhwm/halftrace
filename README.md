# halftrace

**Diagnostics for agent loops. Measure how fast each capability decays over long trajectories.**

[![CI](https://github.com/ruairidhwm/halftrace/actions/workflows/test.yml/badge.svg)](https://github.com/ruairidhwm/halftrace/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/halftrace.svg)](https://pypi.org/project/halftrace/)
[![Python](https://img.shields.io/pypi/pyversions/halftrace.svg)](https://pypi.org/project/halftrace/)
[![License](https://img.shields.io/pypi/l/halftrace.svg)](LICENSE)

Agents fail in four distinct ways as trajectories get longer: they forget state, drift from instructions, repeat tool calls, and terminate prematurely. Each has a different **halftrace** - the trajectory length at which the behaviour is half-degraded. Existing eval frameworks measure task success at one point and miss all four curves. `halftrace` measures them directly.

## The halftrace concept

A *halftrace* is the trajectory length, in tool calls, at which a given agent capability is degraded by 50% relative to its baseline at low N.

Different capabilities decay at different rates. A model's instruction-following might have a halftrace of 30 tool calls while its state recall has a halftrace of 150, meaning instruction adherence falls off five times faster than memory.

The library measures four halftraces per (agent, model, task):

| Probe | What it measures |
| --- | --- |
| `state_amnesia` | Retention of facts introduced earlier in the trajectory |
| `instruction_decay` | Adherence to system-prompt rules over time |
| `tool_repetition` | Avoidance of re-calling tools with the same arguments |
| `premature_termination` | Completing the task before declaring done |
| `narration_substitution` | Emitting tool calls rather than describing them |

## What this is

A measurement library for agent trajectories. You bring the agent. `halftrace` instruments the trajectory and tells you which capability decays first.

The library was built to answer one question: when an agent fails on a long-horizon task, *which* failure mode caused it? The answer matters because the failure modes have completely different fixes: better prompting, better tools, better memory, or a different model, and you can't choose without measuring them separately.

## What this isn't

- **Not a benchmark.** No leaderboard, no canonical task set. You define the tasks.
- **Not an eval framework.** If you want grading, scoring rubrics, or production observability, use [Inspect](https://inspect.ai-safety-institute.org.uk/), [Braintrust](https://www.braintrust.dev/), or [Langsmith](https://www.langchain.com/langsmith). `halftrace` is a diagnostic instrument that sits alongside them.
- **Not an agent framework.** It doesn't build agents. It measures agents you've already built.

## Install

Once 0.1.0 ships:

```bash
pip install halftrace
```

Optional extras:

```bash
pip install "halftrace[anthropic]"
pip install "halftrace[openai]"
pip install "halftrace[all]"
```

Requires Python 3.11+.

## License

MIT.
