from __future__ import annotations

from typing import Any

from .intent import IntentService
from .memory import SessionMemory
from .models import ParsedAction, PipelineResult
from .stt import STTService
from .tools import LocalToolExecutor


class VoiceAgentPipeline:
    def __init__(self, stt: STTService, intent: IntentService, tools: LocalToolExecutor, memory: SessionMemory):
        self.stt = stt
        self.intent = intent
        self.tools = tools
        self.memory = memory

    def analyze_audio(self, audio_bytes: bytes, source_name: str = "input.wav") -> PipelineResult:
        errors: list[str] = []

        transcript = ""
        raw_intent_response = ""
        actions: list[ParsedAction] = []
        execution_preview: list[dict[str, Any]] = []

        try:
            transcript, mode = self.stt.transcribe(audio_bytes=audio_bytes, source_name=source_name)
            self.memory.add("stt", {"mode": mode, "transcript": transcript})
        except Exception as e:
            errors.append(f"STT failed: {e}")
            return PipelineResult(
                transcript="",
                actions=[],
                execution_preview=[],
                execution_results=[],
                raw_intent_response="",
                errors=errors,
            )

        try:
            actions, raw_intent_response = self.intent.detect_actions(
                transcript=transcript,
                chat_history=self.memory.last_n(12),
            )
            self.memory.add(
                "intent",
                {
                    "transcript": transcript,
                    "actions": [a.__dict__ for a in actions],
                    "raw": raw_intent_response,
                },
            )
        except Exception as e:
            errors.append(f"Intent detection failed: {e}")

        for action in actions:
            try:
                execution_preview.append(self.tools.preview(action))
            except Exception as e:
                execution_preview.append({"intent": action.intent, "error": str(e), "requires_confirmation": False})

        return PipelineResult(
            transcript=transcript,
            actions=actions,
            execution_preview=execution_preview,
            execution_results=[],
            raw_intent_response=raw_intent_response,
            errors=errors,
        )

    def execute_actions(self, actions: list[ParsedAction]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        latest_summary: str | None = None

        for action in actions:
            try:
                result = self.tools.execute(action, summary_result=latest_summary)
                if result.get("action") == "summarize_text" and result.get("status") == "ok":
                    latest_summary = result.get("summary", "")
                results.append(result)
                self.memory.add("action", {"intent": action.intent, "result": result})
            except Exception as e:
                err = {"status": "error", "intent": action.intent, "message": str(e)}
                results.append(err)
                self.memory.add("action", err)

        return results
