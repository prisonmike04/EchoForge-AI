# EchoForge AI — Voice-Controlled Local AI Agent

A local-first voice agent that:
1. Accepts audio from microphone or file upload.
2. Transcribes audio to text.
3. Detects one or more intents (compound command support).
4. Executes local tools safely (restricted to `output/`).
5. Shows full pipeline state in a Streamlit UI.

## Features implemented

- **Audio Input**
  - Microphone recording via Streamlit `audio_input`.
  - Upload existing audio (`.wav`, `.mp3`, `.m4a`, `.ogg`).
- **Speech-to-Text (STT)**
  - Local HuggingFace Whisper (`openai/whisper-small.en`) by default.
  - Optional API fallback to OpenAI STT when local inference fails or is too slow.
- **Intent Understanding (LLM)**
  - Local model via **Ollama** (`llama3.1:8b` by default).
  - Returns structured JSON action plans.
- **Tool Execution**
  - Supported intents:
    - `create_file`
    - `write_code`
    - `summarize_text`
    - `general_chat`
  - `create_file` also supports folder creation when path ends with `/`.
  - **Safety constraint**: file writes are blocked outside `output/`.
- **UI Pipeline visibility**
  - Shows transcript, detected intents, planned actions, confirmation state, and execution output.
- **Important implementations required by prompt**
  - ✅ Compound commands.
  - ✅ Human-in-the-loop confirmation before file operations.
  - ✅ Graceful degradation and clear errors.
  - ✅ Persistent session memory (`output/session_memory.json`).
  - ✅ Benchmark script and model comparison section.

---

## Project structure

- [app.py](app.py) — Streamlit frontend.
- [agent/stt.py](agent/stt.py) — STT service (local + optional fallback).
- [agent/intent.py](agent/intent.py) — Intent planner (Ollama + heuristic fallback).
- [agent/tools.py](agent/tools.py) — Local tool execution and path safety.
- [agent/pipeline.py](agent/pipeline.py) — Orchestration logic.
- [agent/memory.py](agent/memory.py) — Persistent action/chat memory.
- [benchmark_models.py](benchmark_models.py) — Simple latency benchmark.
- [MODEL_BENCHMARK_ARTICLE.md](MODEL_BENCHMARK_ARTICLE.md) — Benchmark write-up template + analysis.
- [output](output) — Safe write target for all generated files.

---

## Setup

### 1) Clone

```bash
git clone https://github.com/prisonmike04/EchoForge-AI.git
cd EchoForge-AI
```

### 2) Python env + dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Ollama (local intent model)
Install Ollama and pull a model:

```bash
ollama pull llama3.1:8b
```

### 4) Environment variables

```bash
cp .env.example .env
```

Edit `.env` if needed.

### 5) Run app

```bash
streamlit run app.py
```

---

## How the architecture works

1. **Input stage**: audio bytes captured from mic or uploaded file.
2. **STT stage** (`STTService`): local Whisper transcribes audio.
3. **Intent stage** (`IntentService`): transcript + recent memory are sent to Ollama to produce a JSON action plan. If this fails, heuristic fallback is used.
4. **Preview stage** (`LocalToolExecutor.preview`): system creates execution previews and flags file ops for confirmation.
5. **Execution stage** (`LocalToolExecutor.execute`): runs confirmed actions.
6. **Memory stage** (`SessionMemory`): stores STT outputs, plans, and results persistently.

---

## Safety design

All file operations resolve paths against `output/` and reject path traversal:
- Allowed: `output/my_notes.txt`
- Rejected: `../../etc/passwd`

---

## Graceful degradation

- If STT fails locally:
  - If `OPENAI_API_KEY` exists, app uses OpenAI transcription fallback.
  - Otherwise a user-readable error is shown.
- If Ollama is unavailable:
  - Heuristic intent planning still enables basic operation.
  - Summarization uses a sentence-based fallback.
  - Code generation uses a safe template fallback.
  - General chat returns a clear recovery message.
- If no speech is detected/unintelligible audio:
  - Clear error appears in UI; no unsafe action is executed.

---

## Benchmarking (model speed/performance)

Run:

```bash
python benchmark_models.py /path/to/sample.wav --runs 3
```

The script prints:
- Average STT latency
- Average intent-planning latency
- Detected actions

### Example comparison template (fill with your machine values)

| Component | Model | Avg Latency | Notes |
|---|---|---:|---|
| STT | `openai/whisper-small.en` | ~X.XXs | Local CPU baseline |
| STT (fallback) | `gpt-4o-mini-transcribe` | ~Y.YYs | Faster on weaker local hardware, requires internet/API |
| Intent | `llama3.1:8b` (Ollama) | ~Z.ZZs | Good structured planning locally |

### Hardware workaround note

If your machine cannot run local Whisper fast enough, enabling OpenAI STT fallback is acceptable in this project. Documented via `.env` + this section.

---

## Compound command example

Voice input:
> “Summarize this text and save it to summary.txt …”

Expected plan:
1. `summarize_text`
2. `create_file` with path `summary.txt`

The execution pipeline auto-passes summary output into the file write step.

---

## Deployment / GitHub deliverable

After local validation, push:

```bash
git add .
git commit -m "Build EchoForge local voice-controlled AI agent"
git push origin main
```

If this repository is new, set remote first:

```bash
git remote add origin https://github.com/prisonmike04/EchoForge-AI.git
git branch -M main
git push -u origin main
```
