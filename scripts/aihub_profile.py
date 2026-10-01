#!/usr/bin/env python3
"""Submit profiling jobs to AI Hub cloud-hosted Snapdragon X Elite devices.

This is how SnapNote obtains HONEST benchmark numbers for the pitch deck:
real measurements from real Snapdragon X Elite hardware, via Qualcomm AI Hub.

The script uses only documented ``qai-hub`` CLI commands (see README.md):

    qai-hub configure --api_token <TOKEN>
    qai-hub list-devices --filter "Snapdragon X Elite"
    qai-hub submit-profile-job --model <exported-model> --device "Snapdragon X Elite CRD"

Prerequisites (see README.md "Snapdragon deployment guide"):
  1. A Qualcomm ID + API token in the QAI_HUB_API_TOKEN environment variable.
  2. Compiled model artifacts from the verified AI Hub Models recipes:
       - whisper_base            (Whisper-Base ASR)
       - llama_v3_2_3b_instruct  (Llama 3.2 3B Instruct)
     Produce them with:
       qai-hub-models export whisper_base --device "Snapdragon X Elite CRD"
       python -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
           --device "Snapdragon X Elite CRD" --output-dir genie_bundle \
           --skip-profiling --skip-inferencing

Usage:
    QAI_HUB_API_TOKEN=... python scripts/aihub_profile.py \
        --whisper-artifact ./whisper_base_qnn.bin \
        --llama-artifact ./genie_bundle

The script prints a Markdown results table for the deck. Latency/power cells
are marked "fill from your AI Hub run" until the jobs complete — nothing is
invented.

NOTE: CLI output parsing below is best-effort; verify flags/output against
the current qai-hub docs (https://workbench.aihub.qualcomm.com/docs/hub/cli.html).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

DEVICE_FILTER = "Snapdragon X Elite"
DEVICE_NAME = "Snapdragon X Elite CRD"

VERIFIED_RECIPES = {
    "whisper": "whisper_base",            # verified: qualcomm/ai-hub-models README
    "llama": "llama_v3_2_3b_instruct",    # verified: qualcomm/ai-hub-models tutorial
}


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile SnapNote models on hosted Snapdragon X Elite devices.")
    parser.add_argument("--whisper-artifact", default=None,
                        help="Path to the compiled Whisper artifact (from AI Hub export).")
    parser.add_argument("--llama-artifact", default=None,
                        help="Path to the compiled Llama artifact (from AI Hub export).")
    parser.add_argument("--device", default=DEVICE_NAME)
    args = parser.parse_args()

    token = os.environ.get("QAI_HUB_API_TOKEN")
    if not token:
        die(
            "QAI_HUB_API_TOKEN is not set.\n"
            "Get a token at https://workbench.aihub.qualcomm.com "
            "(Account -> Settings -> API Token), then:\n"
            "  export QAI_HUB_API_TOKEN=<your-token>\n"
            "  qai-hub configure --api_token $QAI_HUB_API_TOKEN"
        )
    if not shutil.which("qai-hub"):
        die(
            "The 'qai-hub' CLI was not found.\n"
            "Install it with:  pip install qai-hub\n"
            "Then configure:   qai-hub configure --api_token $QAI_HUB_API_TOKEN"
        )

    # 1. Configure (documented command).
    print("$ qai-hub configure --api_token $QAI_HUB_API_TOKEN")
    cfg = run(["qai-hub", "configure", "--api_token", token])
    if cfg.returncode != 0:
        die(f"'qai-hub configure' failed:\n{cfg.stderr or cfg.stdout}")

    # 2. Confirm the target device exists on AI Hub.
    print(f'$ qai-hub list-devices --filter "{DEVICE_FILTER}"')
    devs = run(["qai-hub", "list-devices", "--filter", DEVICE_FILTER])
    if devs.returncode != 0:
        die(f"'qai-hub list-devices' failed:\n{devs.stderr or devs.stdout}")
    if args.device not in devs.stdout:
        die(
            f'Device "{args.device}" not found in list-devices output.\n'
            f"Available devices:\n{devs.stdout}\n"
            "Pick a device from the list and re-run with --device."
        )
    print(f'Found target device: "{args.device}"')

    # 3. Submit profiling jobs for each provided artifact.
    jobs = []
    targets = [
        ("whisper", VERIFIED_RECIPES["whisper"], args.whisper_artifact),
        ("llama", VERIFIED_RECIPES["llama"], args.llama_artifact),
    ]
    for label, recipe, artifact in targets:
        if not artifact:
            print(f"[skip] {label}: no artifact given (--{label}-artifact). "
                  f"Export '{recipe}' per the README deployment guide first.")
            continue
        if not os.path.exists(artifact):
            print(f"[skip] {label}: artifact not found: {artifact}")
            continue
        cmd = ["qai-hub", "submit-profile-job",
               "--model", artifact, "--device", args.device]
        print("$ " + " ".join(cmd))
        res = run(cmd)
        if res.returncode != 0:
            print(f"[warn] profile job for {label} failed:\n{res.stderr or res.stdout}",
                  file=sys.stderr)
            jobs.append((label, recipe, "(submission failed — see above)"))
        else:
            # NOTE: exact job-id format varies by qai-hub version; paste the
            # real ID from the output above. Verify against current docs.
            jobs.append((label, recipe, res.stdout.strip().splitlines()[-1]
                         if res.stdout.strip() else "(see output above)"))

    # 4. Markdown table for the deck — numbers filled from the AI Hub run.
    print()
    print("## SnapNote benchmark results (Snapdragon X Elite, via Qualcomm AI Hub)")
    print()
    print("| Model | AI Hub recipe | Profile job | Inference latency | Notes |")
    print("|---|---|---|---|---|")
    if not jobs:
        for label, recipe in [("whisper", VERIFIED_RECIPES["whisper"]),
                              ("llama", VERIFIED_RECIPES["llama"])]:
            print(f"| {label} | `{recipe}` | — | fill from your AI Hub run | — |")
    else:
        for label, recipe, job in jobs:
            print(f"| {label} | `{recipe}` | {job} | fill from your AI Hub run | — |")
    print()
    print("When the jobs complete, open them in AI Hub Workbench and paste the")
    print("measured latency / power figures into the table above — do not invent them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
