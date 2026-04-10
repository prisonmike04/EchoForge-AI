from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent.config import OUTPUT_DIR
from agent.intent import IntentService
from agent.memory import SessionMemory
from agent.pipeline import VoiceAgentPipeline
from agent.stt import STTService
from agent.tools import LocalToolExecutor

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

with st.sidebar:
    st.header("Input Options")
    audio_from_mic = st.audio_input("Record from microphone")
    uploaded_file = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg"])

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

            st.markdown("### Execution Results")
            for res in results:
                if res.get("status") == "ok":
                    st.success(res)
                else:
                    st.error(res)

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
