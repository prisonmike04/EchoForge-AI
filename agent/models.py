from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_INTENTS = {
    "create_file",
    "write_code",
    "summarize_text",
    "general_chat",
}


@dataclass
class ParsedAction:
    intent: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    transcript: str
    actions: list[ParsedAction]
    execution_preview: list[dict[str, Any]]
    execution_results: list[dict[str, Any]]
    raw_intent_response: str
    errors: list[str] = field(default_factory=list)
