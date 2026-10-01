#!/usr/bin/env python3
"""Synthesize demo audio from demo/lecture_script.md (optional, for demos).

Prefers espeak-ng, falls back to espeak. If neither TTS engine is installed,
this script prints install instructions and exits non-zero — it will not
produce fake audio.

Usage:
    python scripts/generate_sample_audio.py [--out demo/sample_lecture.wav]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

SCRIPT = "demo/lecture_script.md"


def read_script_text() -> str:
    with open(SCRIPT, encoding="utf-8") as f:
        text = f.read()
    # Drop the markdown header / instructions, keep the lecture body.
    _, _, body = text.partition("---")
    body = re.sub(r"\n{3,}", "\n\n", (body or text).strip())
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize demo lecture audio.")
    parser.add_argument("--out", default="demo/sample_lecture.wav")
    args = parser.parse_args()

    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine:
        print(
            "No TTS engine found (looked for espeak-ng and espeak).\n"
            "Install one to synthesize the demo audio:\n"
            "  Ubuntu/Debian:  sudo apt install espeak-ng\n"
            "  macOS:          brew install espeak-ng\n"
            "  Windows:        choco install espeak-ng   (or use the GUI)\n"
            "Then re-run: python scripts/generate_sample_audio.py",
            file=sys.stderr,
        )
        return 2

    text = read_script_text()
    # ~150-175 wpm keeps a ~190-word script in the 60-90 second range.
    cmd = [engine, "-v", "en", "-s", "165", "-w", args.out, text]
    subprocess.run(cmd, check=True)
    print(f"Wrote {args.out} using {engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
