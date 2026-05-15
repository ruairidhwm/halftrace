"""halftrace: diagnostics for agent loops."""

from halftrace.compare import ProbeComparison, compare_profiles
from halftrace.diagnose import Diagnosis, diagnose
from halftrace.fit import ComplianceProfile, analyse_compliance
from halftrace.ingest import from_anthropic_messages, from_openai_messages
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
    "Diagnosis",
    "FindAndSynthesise",
    "Probe",
    "ProbeComparison",
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
    "compare_profiles",
    "diagnose",
    "find_and_synthesise",
    "from_anthropic_messages",
    "from_openai_messages",
    "instruction_decay",
    "narration_substitution",
    "state_amnesia",
    "tool_repetition",
]
