"""Speech-to-text for SnapNote.

Two execution paths, one interface:

* **Dev path (runs today on any CPU laptop):** ``FasterWhisperBackend`` wraps
  ``faster-whisper``. Real transcription, no NPU involved.
* **NPU path (Snapdragon X Elite / Hexagon NPU):** ``AIHubNPUBackend`` documents
  the intended on-device integration: a compiled QNN/ONNX context binary —
  produced by the Qualcomm AI Hub export flow (see README.md, "Snapdragon
  deployment guide") — loaded through ONNX Runtime with the QNN execution
  provider. It validates artifacts and reports exactly what is missing; it does
  not fake inference.

Nothing here invents performance numbers. Benchmarks come from
``scripts/aihub_profile.py`` against AI Hub cloud-hosted devices.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptionSegment:
    """One timed chunk of a transcript."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Full transcript plus per-segment timing."""

    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    backend: str = ""
    language: str = ""


class TranscriberBackend(abc.ABC):
    """Pluggable transcription backend."""

    name: str = "base"

    @abc.abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe an audio file; raise a clear error if unavailable."""
        raise NotImplementedError


class FasterWhisperBackend(TranscriberBackend):
    """Dev backend: faster-whisper on CPU (or CUDA if present).

    This is the **dev path** — it runs on a laptop CPU today. It is NOT the
    NPU path. Download weights with ``scripts/download_models.py``.
    """

    name = "faster-whisper"

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install the dev dependencies with:\n"
                "    pip install -r requirements.txt\n"
                "then fetch dev weights with:\n"
                "    python scripts/download_models.py --model-size "
                + self.model_size
            ) from exc
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        model = self._load()
        segments_iter, info = model.transcribe(audio_path, language=language)
        segments: List[TranscriptionSegment] = []
        texts: List[str] = []
        for seg in segments_iter:
            text = seg.text.strip()
            segments.append(TranscriptionSegment(start=seg.start, end=seg.end, text=text))
            texts.append(text)
        return TranscriptionResult(
            text=" ".join(texts).strip(),
            segments=segments,
            backend=self.name,
            language=getattr(info, "language", "") or "",
        )


class AIHubNPUBackend(TranscriberBackend):
    """NPU deployment path: compiled Whisper on the Hexagon NPU.

    Intended integration pattern (documented, not executed on a dev machine):

    1. Export a compiled artifact with Qualcomm AI Hub Models, e.g.::

           qai-hub-models export whisper_base --device "Snapdragon X Elite CRD"
           # or fetch a ready-made context binary:
           qai-hub-models fetch Whisper-Base --runtime qnn_context_binary

       (``whisper_base`` is a verified recipe in
       https://github.com/qualcomm/ai-hub-models — see README.md.)
    2. Load the artifact at runtime with ONNX Runtime + the QNN execution
       provider on the Snapdragon X Elite device, and run the model's
       pre/post-processing around it (the same ``demo.py`` recipe that ships
       with ``qai_hub_models.models.whisper_base``).

    What this class implements today:
        * validates that the artifact path exists,
        * lazy-imports ``onnxruntime`` and reports if the QNN execution
          provider is unavailable,
        * refuses to pretend: ``transcribe()`` raises until wired to a real
          compiled artifact on a real device.

    What it does NOT do: invent latencies, fake transcripts, or claim the NPU
    path was executed on hardware this code has never touched.
    """

    name = "aihub-npu"

    def __init__(self, artifact_path: str):
        self.artifact_path = artifact_path
        self._session = None

    def check_prerequisites(self) -> List[str]:
        """Return a list of missing prerequisites (empty = ready to try)."""
        missing: List[str] = []
        if not os.path.isfile(self.artifact_path):
            missing.append(
                f"Compiled artifact not found: {self.artifact_path}\n"
                "  Produce one via the Snapdragon deployment guide in README.md\n"
                "  (qai-hub-models export whisper_base, then fetch/compile)."
            )
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            missing.append("onnxruntime is not installed (pip install onnxruntime).")
        else:
            import onnxruntime as ort

            if "QNNExecutionProvider" not in ort.get_available_providers():
                missing.append(
                    "onnxruntime is installed but the QNN execution provider is "
                    "not available on this machine. On a Snapdragon X Elite "
                    "device, install the onnxruntime build with QNN support."
                )
        return missing

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        missing = self.check_prerequisites()
        if missing:
            raise RuntimeError(
                "AIHubNPUBackend cannot run here — the NPU deployment path "
                "requires a compiled artifact on Snapdragon hardware:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\nSee README.md 'Snapdragon deployment guide' for the export flow."
            )
        # NOTE: full wiring of the Whisper encoder/decoder ONNX graphs through
        # the QNN EP (mel-spectrogram featurization, autoregressive decode loop)
        # is implemented on the target device after AI Hub compilation. Filling
        # this in on a generic Linux VM would be fabrication; it is
        # intentionally left as a documented extension point.
        raise NotImplementedError(
            "The compiled-artifact decode loop is wired on the Snapdragon "
            "device after AI Hub export. Use FasterWhisperBackend for the dev "
            "path; use the deployment guide in README.md for the NPU path."
        )


class Transcriber:
    """Facade: pick a backend by name, transcribe audio."""

    BACKENDS = {
        "faster-whisper": FasterWhisperBackend,
        "aihub-npu": AIHubNPUBackend,
    }

    def __init__(self, backend: str = "faster-whisper", **kwargs):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(self.BACKENDS)}")
        self.backend_name = backend
        self.backend: TranscriberBackend = self.BACKENDS[backend](**kwargs)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        return self.backend.transcribe(audio_path, language=language)
