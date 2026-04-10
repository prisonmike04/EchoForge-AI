# Model Benchmarking Article — EchoForge AI

## Goal
Compare latency and practical behavior of the selected local-first models for:
- Speech-to-Text (STT)
- Intent planning / command decomposition

## Models evaluated

### STT
1. `openai/whisper-small.en` (local HuggingFace)
2. `gpt-4o-mini-transcribe` (API fallback)

### Intent
1. `llama3.1:8b` via Ollama (local)
2. Heuristic fallback (rule-based, no model)

## Method
- Use [benchmark_models.py](benchmark_models.py) with fixed sample audios.
- Run each case 3-5 times.
- Measure mean latency and observe output quality (correct intents, compound command handling).

## Suggested benchmark command

```bash
python benchmark_models.py ./samples/compound_command.wav --runs 3
```

## Results table template

| Task | Model | Mean Latency | Quality Notes |
|---|---|---:|---|
| STT | whisper-small.en (local) | _fill_ | Accurate for clear English, slower on CPU |
| STT | gpt-4o-mini-transcribe (API) | _fill_ | Usually faster on low-end local hardware |
| Intent | llama3.1:8b (Ollama) | _fill_ | Best for JSON plans + compound steps |
| Intent | heuristic fallback | ~0 | Fast but less robust for ambiguous speech |

## Analysis

- **Best local-only stack**: Whisper + Ollama gives offline privacy and full local control.
- **Best low-hardware stack**: API STT + local Ollama balances speed and local intent routing.
- **Compound command behavior**: Ollama planner is better at multi-action decomposition (e.g., summarize then save).

## Final recommendation

Default to local models for privacy and deterministic tool routing. Enable API STT fallback only when local STT latency is unacceptable.
