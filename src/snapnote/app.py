"""Gradio web UI for SnapNote — the dev (CPU) path.

Flow: upload or record audio -> transcribe (faster-whisper, CPU) ->
generate structured notes (extractive backend by default; local LLM optional).

This UI is the **dev path**: it runs on a laptop today. The NPU deployment
path (compiled Whisper + Llama on the Snapdragon X Elite Hexagon NPU via
Qualcomm AI Hub) is documented in README.md and is NOT what this UI executes.

Run:  python -m snapnote.app   (from the repo root, or see README quickstart)
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snapnote.notes import NotesGenerator  # noqa: E402
from snapnote.transcribe import Transcriber  # noqa: E402

_TRANSCRIBER: Transcriber | None = None
_NOTES_GEN: NotesGenerator | None = None


def _get_transcriber(model_size: str) -> Transcriber:
    global _TRANSCRIBER
    if _TRANSCRIBER is None or _TRANSCRIBER.backend.model_size != model_size:
        _TRANSCRIBER = Transcriber(backend="faster-whisper", model_size=model_size)
    return _TRANSCRIBER


def _get_notes_gen() -> NotesGenerator:
    global _NOTES_GEN
    if _NOTES_GEN is None:
        _NOTES_GEN = NotesGenerator(backend="extractive")
    return _NOTES_GEN


def transcribe_audio(audio_path: str | None, model_size: str) -> str:
    """Gradio callback: audio in -> transcript text out (dev path)."""
    if not audio_path:
        return "Upload or record audio first."
    try:
        result = _get_transcriber(model_size).transcribe(audio_path)
    except RuntimeError as exc:
        return f"Transcription unavailable:\n{exc}"
    except FileNotFoundError as exc:
        return f"Audio error: {exc}"
    return result.text or "(no speech detected)"


def generate_notes(transcript: str) -> str:
    """Gradio callback: transcript in -> study-notes Markdown out (dev path)."""
    transcript = (transcript or "").strip()
    if not transcript or transcript.startswith("Upload or record"):
        return "Transcribe audio first, then generate notes."
    try:
        notes = _get_notes_gen().generate(transcript)
    except ValueError as exc:
        return f"Notes error: {exc}"
    return notes.to_markdown()


def save_notes(transcript: str, notes_md: str) -> str:
    """Gradio callback: persist the current transcript + notes to a file."""
    if not notes_md or notes_md.startswith("Transcribe"):
        return "Nothing to save yet — transcribe and generate notes first."
    out_dir = os.path.join(tempfile.gettempdir(), "snapnote")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "snapnote_session.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SnapNote session (dev path)\n\n")
        f.write("## Transcript\n\n" + (transcript or "") + "\n\n")
        f.write(notes_md)
    return f"Saved to {path}"


def build_ui():
    """Construct the Gradio UI. Imported lazily so the module stays light."""
    import gradio as gr

    with gr.Blocks(title="SnapNote") as demo:
        gr.Markdown(
            "# SnapNote — lecture audio to study notes\n"
            "_Dev path (CPU laptop)._ The Snapdragon X Elite NPU deployment is "
            "documented in README.md; this UI does not run it."
        )
        with gr.Row():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                             label="Lecture audio")
            model_size = gr.Dropdown(choices=["tiny", "base", "small"],
                                     value="base", label="Whisper dev model")
        transcribe_btn = gr.Button("1. Transcribe", variant="primary")
        transcript_box = gr.Textbox(label="Transcript", lines=10)
        notes_btn = gr.Button("2. Generate study notes", variant="primary")
        notes_md = gr.Markdown(label="Study notes")
        save_btn = gr.Button("3. Save session")
        save_status = gr.Textbox(label="Save status", interactive=False)

        transcribe_btn.click(transcribe_audio, inputs=[audio, model_size],
                             outputs=transcript_box)
        notes_btn.click(generate_notes, inputs=transcript_box, outputs=notes_md)
        save_btn.click(save_notes, inputs=[transcript_box, notes_md],
                       outputs=save_status)
    return demo


def main() -> None:
    ui = build_ui()
    ui.launch()


if __name__ == "__main__":
    main()
