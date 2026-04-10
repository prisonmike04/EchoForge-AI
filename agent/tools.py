from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, OUTPUT_DIR
from .models import ParsedAction


class LocalToolExecutor:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preview(self, action: ParsedAction) -> dict[str, Any]:
        intent = action.intent
        params = action.params

        if intent in {"create_file", "write_code"}:
            requested = str(params.get("path", "untitled.txt"))
            safe_path = self._safe_output_path(requested)
            return {
                "requires_confirmation": True,
                "intent": intent,
                "requested_path": requested,
                "safe_path": str(safe_path),
                "description": f"Will {intent.replace('_', ' ')} at {safe_path.name}",
            }

        return {
            "requires_confirmation": False,
            "intent": intent,
            "description": f"Will execute {intent.replace('_', ' ')}",
        }

    def execute(self, action: ParsedAction, summary_result: str | None = None) -> dict[str, Any]:
        intent = action.intent
        params = action.params

        if intent == "create_file":
            return self._create_file(params, summary_result=summary_result)
        if intent == "write_code":
            return self._write_code(params)
        if intent == "summarize_text":
            return self._summarize_text(params)
        if intent == "general_chat":
            return self._general_chat(params)

        return {"status": "error", "message": f"Unsupported intent: {intent}"}

    def _safe_output_path(self, path_text: str) -> Path:
        rel = Path(path_text).as_posix().lstrip("/")
        candidate = (self.output_dir / rel).resolve()

        if not str(candidate).startswith(str(self.output_dir)):
            raise ValueError("Unsafe path blocked. Only output/ is writable.")

        return candidate

    def _create_file(self, params: dict[str, Any], summary_result: str | None = None) -> dict[str, Any]:
        target = self._safe_output_path(str(params.get("path", "note.txt")))
        if str(params.get("path", "")).endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            return {
                "status": "ok",
                "action": "create_folder",
                "path": str(target),
            }

        target.parent.mkdir(parents=True, exist_ok=True)

        content = params.get("content", "")
        if summary_result and not content:
            content = summary_result

        target.write_text(str(content), encoding="utf-8")
        return {
            "status": "ok",
            "action": "create_file",
            "path": str(target),
            "bytes_written": len(str(content).encode("utf-8")),
        }

    def _write_code(self, params: dict[str, Any]) -> dict[str, Any]:
        prompt = str(params.get("content", "")).strip()
        target = self._safe_output_path(str(params.get("path", "generated_code.py")))
        target.parent.mkdir(parents=True, exist_ok=True)

        generated = self._generate_code(prompt=prompt, language=str(params.get("language", "python")))
        target.write_text(generated, encoding="utf-8")

        return {
            "status": "ok",
            "action": "write_code",
            "path": str(target),
            "preview": generated[:500],
        }

    def _summarize_text(self, params: dict[str, Any]) -> dict[str, Any]:
        raw = str(params.get("content", "")).strip()
        if not raw:
            return {"status": "error", "action": "summarize_text", "message": "No text to summarize."}

        summary = self._ask_ollama(
            system="You summarize text in 3-5 concise bullet points.",
            user=f"Summarize:\n\n{raw}",
        )
        return {"status": "ok", "action": "summarize_text", "summary": summary}

    def _general_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        msg = str(params.get("message", "")).strip()
        answer = self._ask_ollama(
            system="You are a concise helpful local AI assistant.",
            user=msg,
        )
        return {"status": "ok", "action": "general_chat", "response": answer}

    def _generate_code(self, prompt: str, language: str) -> str:
        return self._ask_ollama(
            system=(
                "You generate code only. Return only raw source code, no markdown fences. "
                "Prefer safe, small, runnable snippets."
            ),
            user=f"Language: {language}\nTask: {prompt}",
        )

    def _ask_ollama(self, system: str, user: str) -> str:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
