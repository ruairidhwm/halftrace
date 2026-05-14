"""halftrace: diagnostics for agent loops."""

from halftrace.fit import Halftrace, fit_halftrace
from halftrace.probes import Probe, Score, state_amnesia
from halftrace.tasks import FindAndSynthesise, Task, ToolResponse, ToolSpec, find_and_synthesise
from halftrace.trajectory import (
    Role,
    ToolCall,
    ToolResult,
    Trajectory,
    Turn,
)

__version__ = "0.0.1"

__all__ = [
    "FindAndSynthesise",
    "Halftrace",
    "Probe",
    "Role",
    "Score",
    "Task",
    "ToolCall",
    "ToolResponse",
    "ToolResult",
    "ToolSpec",
    "Trajectory",
    "Turn",
    "__version__",
    "find_and_synthesise",
    "fit_halftrace",
    "state_amnesia",
]
