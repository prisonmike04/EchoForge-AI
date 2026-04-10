from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL
from .models import ParsedAction, SUPPORTED_INTENTS

SYSTEM_PROMPT = """
You are an intent planner for a local voice-controlled coding assistant.
Return JSON only using this schema:
{
  "actions": [
    {
      "intent": "create_file|write_code|summarize_text|general_chat",
      "params": {
        "path": "optional relative path under output/",
        "content": "optional text payload",
        "language": "optional language",
        "message": "optional text"
      }
    }
  ]
}
Rules:
- You can return multiple actions for compound commands.
- If user asks to summarize and save, output summarize_text action first, then create_file with target path.
- For general questions/chat, use general_chat with message.
- Never produce intents outside supported values.
- Keep params minimal and specific.
""".strip()


class IntentService:
    def detect_actions(self, transcript: str, chat_history: list[dict[str, Any]] | None = None) -> tuple[list[ParsedAction], str]:
        chat_history = chat_history or []

        try:
            response_text = self._ollama_plan(transcript, chat_history)
            actions = self._parse_actions(response_text)
            if actions:
                return actions, response_text
        except Exception:
            pass

        fallback = self._heuristic_plan(transcript)
        return fallback, "heuristic-fallback"

    def _ollama_plan(self, transcript: str, chat_history: list[dict[str, Any]]) -> str:
        history_text = "\n".join(
            f"- {item.get('kind', 'event')}: {json.dumps(item.get('payload', {}), ensure_ascii=False)}"
            for item in chat_history[-8:]
        )
        user_prompt = (
            f"Conversation history:\n{history_text or '- none'}\n\n"
            f"User transcript:\n{transcript}\n\n"
            "Return JSON only."
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def _parse_actions(self, response_text: str) -> list[ParsedAction]:
        blob = response_text.strip()
        match = re.search(r"\{[\s\S]*\}", blob)
        if match:
            blob = match.group(0)

        data = json.loads(blob)
        actions = data.get("actions", [])
        parsed: list[ParsedAction] = []

        for action in actions:
            intent = str(action.get("intent", "")).strip()
            if intent not in SUPPORTED_INTENTS:
                continue
            params = action.get("params", {}) or {}
            if not isinstance(params, dict):
                params = {}
            parsed.append(ParsedAction(intent=intent, params=params))

        return parsed

    def _heuristic_plan(self, transcript: str) -> list[ParsedAction]:
        t = transcript.lower().strip()
        actions: list[ParsedAction] = []

        wants_summary = "summar" in t
        wants_file = any(k in t for k in ["create file", "save", ".py", ".txt", "write to file"])
        wants_code = any(k in t for k in ["code", "function", "script", "class", "implement", "generate"])

        file_name = self._extract_filename(t)

        if wants_summary:
            actions.append(ParsedAction(intent="summarize_text", params={"content": transcript}))

        if wants_code:
            target_path = file_name or "generated_code.py"
            if "create" in t and "file" in t:
                actions.append(
                    ParsedAction(
                        intent="create_file",
                        params={"path": target_path, "content": ""},
                    )
                )
            actions.append(
                ParsedAction(
                    intent="write_code",
                    params={
                        "path": target_path,
                        "content": transcript,
                        "language": "python" if "python" in t else "text",
                    },
                )
            )
        elif wants_file:
            actions.append(
                ParsedAction(
                    intent="create_file",
                    params={"path": file_name or "note.txt", "content": ""},
                )
            )

        if not actions:
            actions.append(ParsedAction(intent="general_chat", params={"message": transcript}))

        return actions

    @staticmethod
    def _extract_filename(text: str) -> str | None:
        m = re.search(r"([a-zA-Z0-9_\-]+\.(?:py|txt|md|json|csv))", text)
        return m.group(1) if m else None
