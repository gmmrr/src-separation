#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVIDIA NeMo Sortformer speaker diarization pipeline.

This script runs NVIDIA Sortformer diarization and writes JSON files describing
who spoke from which second to which second.

Default input:
  - data/test_diarization/IyLqUS7hRvo_std_vocals.wav
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "data/test_diarization/IyLqUS7hRvo_std_vocals.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_diarization/sortformer_outputs"
DEFAULT_MODEL = "nvidia/diar_sortformer_4spk-v1"
DEFAULT_SAMPLE_RATE = 16000

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _fmt_seconds(value: float) -> str:
    return f"{float(value):.2f}"


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _collect_audio_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return files

    raise FileNotFoundError(
        f"Input path does not exist: {path}. "
        "If you are using the default input, check that the file exists under "
        "data/test_diarization/ at the repository root."
    )


def _load_sortformer_model(model_name: str, token: Optional[str], device: str):
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("nemo_toolkit is not installed. Install NVIDIA NeMo before running Sortformer diarization.") from exc

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # NeMo pretrained checkpoints are usually loaded via Hugging Face login.
    # We keep the token path for convenience, but also support the standard
    # `from_pretrained(model_name)` flow when no token is provided.
    try:
        if token:
            try:
                model = SortformerEncLabelModel.from_pretrained(model_name, token=token)
            except TypeError:
                model = SortformerEncLabelModel.from_pretrained(model_name, use_auth_token=token)
        else:
            model = SortformerEncLabelModel.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Sortformer model: {model_name}. "
            "Make sure NeMo is installed and you have accepted the model terms / logged in with "
            "`huggingface-cli login`, or pass --hf-token."
        ) from exc

    if model is None:
        raise RuntimeError(f"Failed to load Sortformer model: {model_name}")

    model = model.to(torch.device(device))
    model.eval()
    return model


def _extract_annotation_rows(raw_output):
    """Normalize NeMo Sortformer output into per-file rows."""
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    if raw_output is None:
        return []

    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        return raw_output

    # Single-file output may come back as a flat list of segments.
    if isinstance(raw_output, list):
        return [raw_output]

    return [[raw_output]]


def _parse_segment(item) -> Optional[Dict]:
    """Parse one Sortformer segment into the shared JSON schema."""
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None

        if text.startswith("[") or text.startswith("("):
            try:
                item = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                item = text
        else:
            item = text

    if isinstance(item, str):
        parts = item.split()
        if len(parts) < 3:
            return None
        start_token, end_token, speaker_token = parts[0], parts[1], parts[2]
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        start_token, end_token, speaker_token = item[0], item[1], item[2]
    else:
        return None

    try:
        start = float(start_token)
        end = float(end_token)
    except (TypeError, ValueError):
        return None

    if end <= start:
        return None

    speaker_text = str(speaker_token).strip()
    speaker_match = re.search(r"(\d+)$", speaker_text)
    if speaker_match:
        speaker_idx = int(speaker_match.group(1))
        speaker = f"SPEAKER_{speaker_idx:02d}"
    else:
        speaker = speaker_text.upper()

    return {
        "speaker": speaker,
        "start": _fmt_seconds(start),
        "end": _fmt_seconds(end),
        "duration": _fmt_seconds(end - start),
    }


def _annotation_to_segments(diarization_output) -> List[Dict]:
    segments: List[Dict] = []
    rows = _extract_annotation_rows(diarization_output)

    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) == 3 and not isinstance(row[0], (list, tuple, dict, str)):
            parsed = _parse_segment(row)
            if parsed is not None:
                segments.append(parsed)
            continue

        if isinstance(row, (list, tuple, set)):
            for item in row:
                parsed = _parse_segment(item)
                if parsed is not None:
                    segments.append(parsed)
        else:
            parsed = _parse_segment(row)
            if parsed is not None:
                segments.append(parsed)

    segments.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return segments


def _find_overlaps(segments: Sequence[Dict]) -> List[Dict]:
    """Find intervals where two or more diarization segments overlap."""
    events = []

    for seg in segments:
        events.append((seg["start"], "start", seg["speaker"]))
        events.append((seg["end"], "end", seg["speaker"]))

    events.sort(key=lambda item: (item[0], 0 if item[1] == "end" else 1))

    overlaps = []
    active = set()
    previous_time = None

    for time, event_type, speaker in events:
        if previous_time is not None and time > previous_time and len(active) >= 2:
            overlaps.append(
                {
                    "start": _fmt_seconds(previous_time),
                    "end": _fmt_seconds(time),
                    "duration": _fmt_seconds(time - previous_time),
                    "speakers": sorted(active),
                    "num_speakers": len(active),
                }
            )

        if event_type == "start":
            active.add(speaker)
        else:
            active.discard(speaker)

        previous_time = time

    return overlaps


def diarize_single(
    model,
    input_path: Path,
    output_dir: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    postprocessing_yaml: Optional[Path] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    force: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / f"{input_path.stem}_diarization.json"
    if out_json.exists() and not force:
        return out_json

    diarization_output = model.diarize(
        audio=[str(input_path)],
        sample_rate=sample_rate,
        batch_size=batch_size,
        include_tensor_outputs=False,
        postprocessing_yaml=str(postprocessing_yaml) if postprocessing_yaml else None,
        num_workers=num_workers,
        verbose=verbose,
    )

    segments = _annotation_to_segments(diarization_output)
    overlaps = _find_overlaps(segments)
    speakers = sorted({seg["speaker"] for seg in segments})

    result = {
        "input": str(input_path.resolve()),
        "num_speakers_detected": len(speakers),
        "speakers": speakers,
        "segments": segments,
        "overlaps": overlaps,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_json


def run_diarization(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
    hf_token: Optional[str] = None,
    device: str = "auto",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    postprocessing_yaml: Optional[Path | str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    force: bool = False,
) -> List[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    postprocessing_yaml = Path(postprocessing_yaml).expanduser().resolve() if postprocessing_yaml else None

    used_device = _resolve_device(device)
    model = _load_sortformer_model(
        model_name=model_name,
        token=hf_token,
        device=used_device,
    )

    input_files = _collect_audio_files(input_path)

    outputs = []
    for in_file in input_files:
        out_json = diarize_single(
            model=model,
            input_path=in_file,
            output_dir=output_dir,
            sample_rate=sample_rate,
            postprocessing_yaml=postprocessing_yaml,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            force=force,
        )
        outputs.append(out_json)

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NVIDIA Sortformer speaker diarization and output JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input audio file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where diarization JSON files will be written.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="NeMo Sortformer model name.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token. If omitted, HF_TOKEN will be used.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate passed to NeMo diarize().",
    )
    parser.add_argument(
        "--postprocessing-yaml",
        type=Path,
        default=None,
        help="Optional NeMo post-processing YAML file for better timestamp extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used by NeMo diarize().",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers used by NeMo diarize().",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show diarization progress output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSON outputs.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    outputs = run_diarization(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model_name,
        hf_token=args.hf_token,
        device=args.device,
        sample_rate=args.sample_rate,
        postprocessing_yaml=args.postprocessing_yaml,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose,
        force=args.force,
    )

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
