from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path

import librosa
import numpy as np
import requests
from transformers import pipeline

from .config import OPENAI_API_KEY, OPENAI_STT_MODEL, WHISPER_MODEL


class STTService:
    def __init__(self) -> None:
        self._asr = None

    def _load_local_model(self):
        if self._asr is None:
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL,
                chunk_length_s=20,
                device="cpu",
            )
        return self._asr

    def transcribe(self, audio_bytes: bytes, source_name: str = "input.wav") -> tuple[str, str]:
        """
        Returns: (transcript, mode)
        mode is either 'local' or 'api-fallback'.
        """
        if not audio_bytes:
            raise ValueError("No audio received.")

        try:
            return self._transcribe_local(audio_bytes, source_name=source_name), "local"
        except Exception as local_err:
            if OPENAI_API_KEY:
                try:
                    return self._transcribe_openai(audio_bytes, source_name), "api-fallback"
                except Exception as api_err:
                    raise RuntimeError(
                        f"Local STT failed ({local_err}) and API fallback failed ({api_err})."
                    ) from api_err
            raise RuntimeError(f"Local STT failed and no API fallback configured: {local_err}")

    def _transcribe_local(self, audio_bytes: bytes, source_name: str = "input.wav") -> str:
        asr = self._load_local_model()
        suffix = Path(source_name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            waveform, sample_rate = librosa.load(Path(f.name).as_posix(), sr=16000, mono=True)

        if waveform.size == 0:
            raise ValueError("Audio appears empty or unreadable.")
        if self._is_effectively_silent(waveform):
            raise ValueError("No speech detected in audio. Please speak and try again.")

        result = asr({"raw": np.asarray(waveform), "sampling_rate": sample_rate})
        text = result.get("text", "").strip()
        if self._is_invalid_transcript(text):
            raise ValueError("No clear speech detected in audio. Please try again.")
        return text

    def _transcribe_openai(self, audio_bytes: bytes, source_name: str) -> str:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (source_name, io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": OPENAI_STT_MODEL},
            timeout=120,
        )
        response.raise_for_status()
        text = response.json().get("text", "").strip()
        if self._is_invalid_transcript(text):
            raise ValueError("No clear speech detected in audio (API STT). Please try again.")
        return text

    @staticmethod
    def _is_effectively_silent(waveform: np.ndarray) -> bool:
        if waveform.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        peak = float(np.max(np.abs(waveform)))
        return rms < 5e-4 and peak < 8e-3

    @staticmethod
    def _is_invalid_transcript(text: str) -> bool:
        if not text:
            return True
        cleaned = text.strip()
        if cleaned in {".", "..", "...", "!", "?", "-", "_"}:
            return True
        alnum = re.sub(r"[^A-Za-z0-9]+", "", cleaned)
        return len(alnum) == 0
