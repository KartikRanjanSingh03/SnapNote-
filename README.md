# SnapNote

**On-device AI lecture transcription + smart study notes, built for the
Snapdragon X Elite NPU in HP AI PCs.**

SnapNote turns lecture audio into a transcript plus structured study notes
(key points, definitions, action items, quiz questions) — 100% on-device, no
cloud, no recording ever leaving the laptop. Transcription uses a
Whisper-class model; notes use a small Llama-class model. Both models have a
documented Qualcomm AI Hub quantization/compilation path to the Hexagon NPU.

> **Honesty note (read first).** This repo has two execution paths and never
> confuses them:
>
> | Path | Runs where | Status |
> |---|---|---|
> | **Dev path** (`faster-whisper` + extractive/LLM notes on CPU) | Any laptop, today | ✅ Runs — see Quickstart |
> | **NPU path** (compiled Whisper + Llama on Hexagon NPU) | Snapdragon X Elite device / AI Hub cloud-hosted devices | 📝 documented + scripted, **not executed on this repo's dev machine** |
>
> There are **no performance numbers** anywhere in this repo — no latency,
> TOPS, battery %, or accuracy figures. Where a number belongs, the code and
> docs point at `scripts/aihub_profile.py` and say *"fill from your AI Hub
> run"*. Any number you see in the pitch deck must come from a real AI Hub
> profiling run, or it must not be there.

---

## Architecture

```
                        ┌────────────────────────────────────────────┐
                        │                SnapNote                     │
                        │                                            │
  lecture audio ──▶ ┌───┴───┐   transcript   ┌─────────┴─────────┐    │
  (upload / mic)   │ TRAN- │ ─────────────▶ │  NOTES GENERATOR  │    │
                   │ SCRIBE │                │                   │    │
                   └───┬───┘                └─────────┬─────────┘    │
                       │                              │             │
        ┌──────────────┴──────────────┐  ┌────────────┴────────────┐ │
        │ DEV PATH (runs today, CPU)  │  │ NPU PATH (Snapdragon    │ │
        │ faster-whisper (tiny/base)  │  │ X Elite, Hexagon NPU)   │ │
        │ extractive summarizer       │  │                         │ │
        │ (or local transformers LLM) │  │ Whisper-Base + Llama    │ │
        └──────────────┬──────────────┘  │ 3.2 3B, quantized via   │ │
                       │                 │ Qualcomm AI Hub, run    │ │
                       ▼                 │ via QNN EP / Genie      │ │
              ┌────────────────┐         └────────────┬────────────┘ │
              │  Gradio web UI │                      │             │
              │  src/snapnote  │◀─────────────────────┘             │
              │  /app.py       │   same interfaces, swapped backends │
              └────────────────┘                                   │
                        └────────────────────────────────────────────┘

  transcript + notes ──▶ Markdown file / on-screen (never leaves device)
```

The two paths share interfaces (`Transcriber`/`TranscriberBackend`,
`NotesGenerator`), so moving from dev to NPU is a backend swap, not a
rewrite.

## Verified Qualcomm AI Hub model IDs

Every AI Hub model ID referenced in this repo was verified against the real
[qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models) repository
on 2026-09-30:

| Recipe ID (CLI) | Display name | Verified via |
|---|---|---|
| `whisper_base` | Whisper-Base | [`src/qai_hub_models/models/whisper_base/README.md`](https://github.com/qualcomm/ai-hub-models/blob/HEAD/src/qai_hub_models/models/whisper_base/README.md) — install/demo/export commands use this exact ID |
| `whisper_small`, `whisper_medium`, `whisper_small_quantized`, `whisper_medium_quantized` | Whisper-Small / -Medium / -Small-Quantized / -Medium-Quantized | Model READMEs exist under `src/qai_hub_models/models/` in the same repo |
| `llama_v3_2_3b_instruct` | Llama 3.2 3B Instruct | [`tutorials/llm/quantize_llama3.md`](https://github.com/qualcomm/ai-hub-models/blob/HEAD/tutorials/llm/quantize_llama3.md) in the official repo (`python -m qai_hub_models.models.llama_v3_2_3b_instruct.export`), plus the [ai-hub-apps chatapp README](https://github.com/qualcomm/ai-hub-apps/blob/HEAD/chatapp_windows_cpp/README.md) exporting this exact model for `"Snapdragon X Elite CRD"` |

If an ID ever stops resolving, run `qai-hub-models info <model-id>` — the
documented way to inspect a model.

## Quickstart (dev path — runs today on a laptop CPU)

```bash
git clone <this-repo> && cd snapnote   # repo root = this directory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch small DEV weights (CPU only — not NPU artifacts)
python scripts/download_models.py --model-size base

# 2. Launch the UI: upload/record audio -> transcript -> study notes
PYTHONPATH=src python -m snapnote.app
# or: PYTHONPATH=src python src/snapnote/app.py
```

Programmatic use:

```python
from snapnote.transcribe import Transcriber
from snapnote.notes import NotesGenerator

text = Transcriber(backend="faster-whisper", model_size="base").transcribe("lecture.wav").text
notes = NotesGenerator(backend="extractive").generate(text)
print(notes.to_markdown())
```

Notes backends: `"extractive"` (default; pure Python, offline, always works —
a **dev fallback**, not an LLM) and `"llm"` (local Llama-class model via
`transformers`; needs an explicit Hugging Face model id and a weight download
— also **dev path only**).

## Snapdragon deployment guide (NPU path)

This path is real, documented, and scripted — it runs on Snapdragon X Elite
hardware / AI Hub cloud-hosted devices, not on a generic Linux VM.

### 0. Prerequisites

- A Qualcomm ID and API token: sign in at
  https://workbench.aihub.qualcomm.com → Account → Settings → API Token.
- Python 3.10–3.13, `pip install qai-hub qai-hub-models`.

### 1. Configure + find the device

```bash
qai-hub configure --api_token $QAI_HUB_API_TOKEN
qai-hub list-devices --filter "Snapdragon X Elite"
# expect a device named "Snapdragon X Elite CRD" (HP AI PCs ship this SoC)
```

### 2. Export the models for on-device deployment

```bash
# Whisper (transcription). Verified recipe: whisper_base
qai-hub-models install whisper_base
qai-hub-models export whisper_base --device "Snapdragon X Elite CRD"
# ..or fetch a ready-made context binary:
qai-hub-models fetch Whisper-Base --runtime qnn_context_binary
# (check available precisions with: qai-hub-models info Whisper-Base)

# Llama 3.2 3B (notes). Verified recipe: llama_v3_2_3b_instruct
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
    --device "Snapdragon X Elite CRD" \
    --output-dir genie_bundle \
    --skip-profiling --skip-inferencing
```

### 3. Quantize

- Whisper: prefer the `whisper_small_quantized` / `whisper_medium_quantized`
  recipes (w8a16, verified in ai-hub-models), or quantize your export with
  AIMET per the model README. Confirm device support with
  `qai-hub-models info Whisper-Small-Quantized` before committing.
- Llama: `python -m qai_hub_models.models.llama_v3_2_3b_instruct.quantize
  --checkpoint meta-llama/Llama-3.2-3B-Instruct --output-dir ./quantized_model`
  (see `tutorials/llm/quantize_llama3.md` in ai-hub-models).

### 4. Compile + profile on the hosted device

```bash
qai-hub submit-profile-job \
    --model <exported-model> \
    --device "Snapdragon X Elite CRD"
```

Or run the scripted version (reads `QAI_HUB_API_TOKEN` from the environment,
checks the device exists, submits jobs, prints the deck table):

```bash
QAI_HUB_API_TOKEN=... python scripts/aihub_profile.py \
    --whisper-artifact ./whisper_base_qnn.bin \
    --llama-artifact ./genie_bundle
```

(A bash equivalent lives at `profile_on_ai_hub.sh` in the repo root — same
documented commands, takes compiled model paths as arguments.)

### 5. Run on-device

- Whisper: load the compiled context binary via ONNX Runtime with the QNN
  execution provider — see `AIHubNPUBackend` in `src/snapnote/transcribe.py`
  for the integration pattern (validates artifacts, refuses to fake results).
  The `whisper_base` AI Hub README additionally notes compatibility with the
  Qualcomm Voice AI SDK.
- Llama: deploy the `genie_bundle` with the Genie runtime per the
  [LLM on Genie tutorial](https://github.com/qualcomm/ai-hub-apps/tree/release/tutorials/llm_on_genie),
  reusing the prompt templates in `src/snapnote/notes.py`
  (`NOTES_SYSTEM_PROMPT`, `NOTES_USER_TEMPLATE`).

## How to obtain benchmark numbers (for the pitch deck)

1. Complete the deployment guide above through step 4.
2. Run `scripts/aihub_profile.py` — it prints a Markdown table; latency/power
   cells read **"fill from your AI Hub run"**.
3. Open the finished jobs in AI Hub Workbench, copy the measured figures into
   the table, paste the table into the deck.
4. **Rule: no number goes in the deck unless it came from that run.** The repo
   contains zero placeholder metrics by design.

## Demo instructions

No microphone handy? Use the bundled demo materials:

```bash
# 1. (optional) synthesize ~60-90s of demo audio from the lecture script
#    needs espeak-ng or espeak; otherwise prints install instructions
python scripts/generate_sample_audio.py
# -> demo/sample_lecture.wav

# 2. inspect the reference transcript + notes (SAMPLE OUTPUT — illustration only)
cat demo/sample_transcript.txt
cat demo/sample_notes.md

# 3. regenerate the notes yourself with the real extractive backend:
PYTHONPATH=src python - <<'EOF'
from snapnote.notes import NotesGenerator
text = open("demo/sample_transcript.txt").read().split("not a real lecture recording.")[-1]
print(NotesGenerator(backend="extractive").generate(text).to_markdown())
EOF

# 4. full flow in the browser UI:
PYTHONPATH=src python -m snapnote.app   # upload demo/sample_lecture.wav
```

`demo/sample_transcript.txt` and `demo/sample_notes.md` are header-labeled
**"SAMPLE OUTPUT — generated from demo script for illustration, not a real
lecture recording"**. They are not benchmark data and must not be presented
as measured results.

## Repo layout

```
snapnote/
├── README.md                    # you are here
├── LICENSE                      # MIT © 2026 Kartik Ranjan Singh
├── requirements.txt
├── .gitignore
├── src/snapnote/
│   ├── __init__.py              # version
│   ├── transcribe.py            # Transcriber: faster-whisper (dev) + AIHubNPUBackend (NPU pattern)
│   ├── notes.py                 # NotesGenerator: extractive (dev fallback) + llm (dev) backends
│   └── app.py                   # Gradio UI (dev path)
├── scripts/
│   ├── aihub_profile.py         # submit AI Hub profile jobs; prints deck table
│   ├── download_models.py       # fetch small DEV weights (CPU, not NPU artifacts)
│   └── generate_sample_audio.py # TTS-synthesize demo audio (espeak-ng/espeak)
└── demo/
    ├── lecture_script.md        # ~190-word photosynthesis lecture script
    ├── sample_transcript.txt    # SAMPLE OUTPUT (illustration only)
    └── sample_notes.md          # SAMPLE OUTPUT (illustration only)
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Kartik Ranjan Singh.
