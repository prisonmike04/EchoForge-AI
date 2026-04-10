from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from agent.config import OUTPUT_DIR
from agent.intent import IntentService
from agent.memory import SessionMemory
from agent.pipeline import VoiceAgentPipeline
from agent.stt import STTService
from agent.tools import LocalToolExecutor


def _guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".md": "markdown",
        ".json": "json",
        ".txt": "text",
    }.get(ext, "text")


def _result_explanation(res: dict[str, Any]) -> str:
    if res.get("status") != "ok":
        return f"Execution failed: {res.get('message', 'Unknown error')}"

    action = res.get("action", "")
    if action == "write_code":
        return "Code generation succeeded and the code has been saved to the target file."
    if action == "create_file":
        note = res.get("note")
        if note:
            return f"File operation completed. {note}"
        return "File was created successfully in the output folder."
    if action == "create_folder":
        return "Folder was created successfully in the output folder."
    if action == "summarize_text":
        return "Text was summarized successfully."
    if action == "general_chat":
        return "Chat response generated successfully."
    return "Action completed successfully."


def _render_execution_results(results: list[dict[str, Any]]) -> None:
    st.markdown("### Execution Results")
    if not results:
        st.info("No execution results yet.")
        return

    for idx, res in enumerate(results, start=1):
        with st.expander(f"Result {idx}: {res.get('action', res.get('intent', 'action'))}", expanded=True):
            if res.get("status") == "ok":
                st.success("Status: ok")
            else:
                st.error("Status: error")

            st.write(_result_explanation(res))

            if res.get("action") == "summarize_text":
                st.markdown("**Summary Output**")
                st.write(res.get("summary", ""))

            if res.get("action") == "general_chat":
                st.markdown("**Assistant Response**")
                st.write(res.get("response", ""))

            path_text = res.get("path")
            if path_text:
                file_path = Path(path_text)
                st.markdown(f"**Saved Path**: {file_path}")
                if file_path.exists() and file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        st.markdown("**File Content**")
                        st.code(content, language=_guess_language(file_path))

                        mirror_dir = OUTPUT_DIR / "downloads"
                        mirror_dir.mkdir(parents=True, exist_ok=True)
                        mirror_path = mirror_dir / file_path.name
                        mirror_path.write_text(content, encoding="utf-8")
                        st.caption(f"Workspace copy saved at: {mirror_path}")

                        st.download_button(
                            label=f"Download {file_path.name}",
                            data=content,
                            file_name=file_path.name,
                            mime="text/plain",
                            key=f"download_{idx}_{file_path.name}",
                        )
                    except Exception as e:
                        st.warning(f"Could not read file content: {e}")

            if res.get("preview"):
                st.markdown("**Generated Preview**")
                st.code(str(res.get("preview")), language="text")

            st.caption(str(res))

st.set_page_config(page_title="EchoForge AI", page_icon="🎙️", layout="wide")
st.title("🎙️ EchoForge AI - Voice-Controlled Local Agent")
st.caption("Speech → Intent(s) → Tool Execution (safe to output/ only)")

memory = SessionMemory(OUTPUT_DIR / "session_memory.json")
agent = VoiceAgentPipeline(
    stt=STTService(),
    intent=IntentService(),
    tools=LocalToolExecutor(output_dir=OUTPUT_DIR),
    memory=memory,
)

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "pending_actions" not in st.session_state:
    st.session_state.pending_actions = []
if "execution_results" not in st.session_state:
    st.session_state.execution_results = []

with st.sidebar:
    st.header("Input Options")
    audio_from_mic = st.audio_input("Record from microphone")
    uploaded_file = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg"])

    with st.expander("Mic troubleshooting", expanded=False):
        st.write("If recording fails with a browser error, try:")
        st.write("1) Open the app using http://localhost:8501")
        st.write("2) Allow microphone permission for localhost in your browser")
        st.write("3) Enable mic for your browser in macOS Privacy settings")
        st.write("4) Close apps using your mic (Zoom/Meet/Teams)")
        st.write("5) Use audio upload as fallback (.wav/.mp3/.m4a/.ogg)")

    st.divider()
    st.subheader("Safety")
    st.write("All file operations are restricted to the output folder.")
    st.write(f"Output path: {OUTPUT_DIR}")

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("1) Transcription + Intent")
    analyze_clicked = st.button("Analyze Audio", type="primary", use_container_width=True)

with col_b:
    st.subheader("2) Execute Planned Actions")
    execute_clicked = st.button("Execute Confirmed Actions", use_container_width=True)

input_bytes = None
source_name = "audio.wav"

if audio_from_mic is not None:
    input_bytes = audio_from_mic.read()
    source_name = "microphone.wav"
elif uploaded_file is not None:
    input_bytes = uploaded_file.read()
    source_name = uploaded_file.name

if analyze_clicked:
    if not input_bytes:
        st.error("Please record or upload an audio file first.")
    else:
        with st.spinner("Running STT + intent classification..."):
            analysis = agent.analyze_audio(input_bytes, source_name=source_name)
            st.session_state.analysis = analysis
            st.session_state.pending_actions = analysis.actions
            st.session_state.execution_results = []

analysis = st.session_state.analysis

if analysis:
    if analysis.errors:
        for err in analysis.errors:
            st.error(err)

    st.markdown("### Transcribed Text")
    st.write(analysis.transcript or "(none)")

    st.markdown("### Detected Intents")
    if not analysis.actions:
        st.warning("No actions detected. The input may be unclear.")
    else:
        for i, action in enumerate(analysis.actions, start=1):
            st.write(f"{i}. {action.intent} — {action.params}")

    st.markdown("### Planned Actions (Pre-Execution)")
    needs_confirmation = False
    confirmed_count = 0

    for i, preview in enumerate(analysis.execution_preview, start=1):
        st.info(f"{i}. {preview.get('description', 'No preview available')}")
        if preview.get("requires_confirmation"):
            needs_confirmation = True
            key = f"confirm_action_{i}"
            confirmed = st.checkbox(
                f"Confirm action {i} ({preview.get('intent')})",
                key=key,
            )
            if confirmed:
                confirmed_count += 1

    if needs_confirmation and confirmed_count == 0:
        st.warning("Confirm at least one file operation to execute.")

if execute_clicked:
    if not st.session_state.pending_actions:
        st.error("No planned actions available. Click Analyze Audio first.")
    else:
        actions_to_run = []
        for i, action in enumerate(st.session_state.pending_actions, start=1):
            requires_confirmation = action.intent in {"create_file", "write_code"}
            if not requires_confirmation:
                actions_to_run.append(action)
                continue

            if st.session_state.get(f"confirm_action_{i}", False):
                actions_to_run.append(action)

        if not actions_to_run:
            st.warning("No actions approved for execution.")
        else:
            with st.spinner("Executing actions..."):
                results = agent.execute_actions(actions_to_run)
            st.session_state.execution_results = results

if st.session_state.execution_results:
    _render_execution_results(st.session_state.execution_results)

st.divider()
st.subheader("Session Memory")
history = memory.list()
if not history:
    st.write("No history yet.")
else:
    for item in reversed(history[-15:]):
        st.write(item)

st.divider()
st.subheader("Graceful Degradation Notes")
st.write(
    "If local Whisper/Ollama is unavailable, the app returns clear errors. "
    "If OPENAI_API_KEY is provided, STT can automatically fall back to API mode."
)

st.subheader("Suggested sample voice commands")
st.write("- Create a Python file named retry.py with a retry decorator function.")
st.write("- Summarize this text and save it to summary.txt: <your text>")
st.write("- Explain what memoization is.")
