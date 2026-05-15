"""halftrace: diagnostics for agent loops."""

from halftrace.fit import ComplianceProfile, analyse_compliance
from halftrace.probes import (
    Probe,
    Score,
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
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
    "ComplianceProfile",
    "FindAndSynthesise",
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
    "analyse_compliance",
    "find_and_synthesise",
    "instruction_decay",
    "narration_substitution",
    "state_amnesia",
    "tool_repetition",
]
