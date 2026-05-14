"""Tasks specify what an agent must accomplish and respond to its tool calls."""

from halftrace.tasks.base import Task, ToolResponse, ToolSpec
from halftrace.tasks.find_and_synthesise import FindAndSynthesise, find_and_synthesise

__all__ = ["FindAndSynthesise", "Task", "ToolResponse", "ToolSpec", "find_and_synthesise"]
