"""halftrace: diagnostics for agent loops."""

from halftrace.fit import Halftrace, fit_halftrace
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
    "Halftrace",
    "Probe",
    "Role",
    "Score",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "Turn",
    "__version__",
    "fit_halftrace",
    "state_amnesia",
]
