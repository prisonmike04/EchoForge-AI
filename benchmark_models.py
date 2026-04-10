from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent.intent import IntentService
from agent.stt import STTService


def benchmark(audio_path: Path, runs: int = 1) -> None:
    stt = STTService()
    intent = IntentService()

    audio_bytes = audio_path.read_bytes()

    stt_times = []
    text = ""
    for _ in range(runs):
        t0 = time.perf_counter()
        text, mode = stt.transcribe(audio_bytes, source_name=audio_path.name)
        stt_times.append(time.perf_counter() - t0)

    intent_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        actions, raw = intent.detect_actions(text)
        intent_times.append(time.perf_counter() - t0)

    print("=== Benchmark Report ===")
    print(f"Audio file: {audio_path}")
    print(f"Runs: {runs}")
    print(f"STT mode: {mode}")
    print(f"Avg STT time: {sum(stt_times)/len(stt_times):.3f}s")
    print(f"Avg Intent time: {sum(intent_times)/len(intent_times):.3f}s")
    print(f"Transcript: {text}")
    print(f"Actions: {[a.intent for a in actions]}")
    print(f"Raw planner output: {raw[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark local STT and intent models.")
    parser.add_argument("audio", type=str, help="Path to test audio file")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    benchmark(Path(args.audio), runs=args.runs)
