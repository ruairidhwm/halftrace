"""halftrace: diagnostics for agent loops."""

from halftrace.probes import Probe, Score, state_amnesia
from halftrace.trajectory import (
    Role,
    ToolCall,
    ToolResult,
    Trajectory,
    Turn,
)

__version__ = "0.0.1"

__all__ = [
    "Probe",
    "Role",
    "Score",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "Turn",
    "__version__",
    "state_amnesia",
]
