"""halftrace: diagnostics for agent loops."""

from halftrace.compare import ProbeComparison, compare_profiles
from halftrace.diagnose import Diagnosis, diagnose
from halftrace.fit import ComplianceProfile, analyse_compliance
from halftrace.ingest import (
    from_anthropic_messages,
    from_langsmith_run,
    from_openai_messages,
)
from halftrace.probes import (
    Probe,
    Score,
    instruction_decay,
    narration_substitution,
    premature_termination,
    state_amnesia,
    tool_repetition,
)
from halftrace.tasks import (
    FindAndSynthesise,
    FindMax,
    Task,
    ToolResponse,
    ToolSpec,
    find_and_synthesise,
    find_max,
)
from halftrace.trajectory import (
    Role,
    ToolCall,
    ToolResult,
    Trajectory,
    Turn,
)

__version__ = "0.1.1"

__all__ = [
    "ComplianceProfile",
    "Diagnosis",
    "FindAndSynthesise",
    "FindMax",
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
    "find_max",
    "from_anthropic_messages",
    "from_langsmith_run",
    "from_openai_messages",
    "instruction_decay",
    "narration_substitution",
    "premature_termination",
    "state_amnesia",
    "tool_repetition",
]
