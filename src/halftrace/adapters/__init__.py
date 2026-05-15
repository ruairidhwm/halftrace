"""Adapters drive an agent through a Task and produce a Trajectory.

Each adapter imports its provider SDK at module load, so importing
`halftrace.adapters.anthropic_adapter` will fail with a clear ModuleNotFoundError
if the corresponding optional extra is not installed.
"""

from halftrace.adapters.anthropic_adapter import (
    DEFAULT_ANTHROPIC_MODEL,
    run_anthropic_task,
)
from halftrace.adapters.openai_adapter import (
    DEFAULT_OPENAI_MODEL,
    run_openai_task,
)

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "run_anthropic_task",
    "run_openai_task",
]
