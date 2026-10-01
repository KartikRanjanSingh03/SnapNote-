#!/usr/bin/env bash
# SnapNote — Qualcomm AI Hub profiling script
#
# Submits profile jobs for the compiled SnapNote models to cloud-hosted
# Snapdragon X Elite devices via the qai-hub CLI, and prints the job URLs.
# Results from these runs are used to fill the benchmark table
# (latency, memory, power) with measured numbers — nothing here is estimated.
#
# Prerequisites:
#   pip install qai-hub qai-hub-models
#   qai-hub configure --api_token <YOUR_TOKEN>   # from https://aihub.qualcomm.com
#
# Usage:
#   ./profile_on_ai_hub.sh <compiled-model-1> [<compiled-model-2> ...]
# Example:
#   ./profile_on_ai_hub.sh whisper_stt.qnn.context.bin notes_llm.qnn.context.bin
#
set -euo pipefail

DEVICE="${SNAPNOTE_DEVICE:-Snapdragon X Elite CRD}"

echo "Listing available Snapdragon X Elite devices..."
qai-hub list-devices --filter "Snapdragon X Elite"

echo
echo "Submitting profile jobs to device: ${DEVICE}"
for model in "$@"; do
  echo "--- Profiling: ${model}"
  qai-hub submit-profile-job --model "${model}" --device "${DEVICE}"
done

echo
echo "Done. Monitor job status and fetch results in the AI Hub Workbench."
