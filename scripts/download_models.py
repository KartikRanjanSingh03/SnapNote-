#!/usr/bin/env python3
"""Download small DEV models for local (CPU) development.

These are **dev models only** — they run on a laptop CPU via faster-whisper
and are NOT the NPU artifacts. The NPU artifacts are compiled context binaries
produced by the Qualcomm AI Hub export flow documented in README.md
("Snapdragon deployment guide").

Usage:
    python scripts/download_models.py [--model-size base] [--model-dir models/]
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Download faster-whisper dev models.")
    parser.add_argument("--model-size", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="faster-whisper model size (dev only)")
    parser.add_argument("--model-dir", default="models",
                        help="Cache directory for downloaded weights")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "faster-whisper is not installed. Install dev dependencies first:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    os.makedirs(args.model_dir, exist_ok=True)
    print(f"Downloading faster-whisper '{args.model_size}' (dev model, CPU) "
          f"into {args.model_dir}/ ...")
    WhisperModel(args.model_size, device="cpu", compute_type="int8",
                 download_root=args.model_dir)
    print("Done. These weights are for local development only — "
          "they are not NPU artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
