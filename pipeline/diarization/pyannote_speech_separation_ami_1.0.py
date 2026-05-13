#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pyannote speaker diarization pipeline.

This script now only runs diarization. It no longer performs speech
separation or writes separated WAV files.

Default demo input:
  - data/test_diarization/IyLqUS7hRvo_std_vocals.wav
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import get_token as hf_get_token


if not hasattr(torchaudio, "AudioMetaData"):
    class _AudioMetaData:
        """Compatibility shim for pyannote.audio import-time type hints."""

        def __init__(
            self,
            num_frames: int,
            sample_rate: int,
            num_channels: int = 1,
            bits_per_sample: int = 16,
            encoding: str = "PCM_S",
        ) -> None:
            self.num_frames = num_frames
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = _AudioMetaData  # type: ignore[attr-defined]

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

if not hasattr(torchaudio, "info"):
    def _info(audio_file, backend=None):
        del backend
        info = sf.info(str(audio_file))
        return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
            num_frames=info.frames,
            sample_rate=info.samplerate,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding=getattr(info, "subtype", "PCM_S"),
        )

    torchaudio.info = _info  # type: ignore[attr-defined]


def _register_safe_globals() -> None:
    """Allow pyannote checkpoints to load under PyTorch 2.6+ weights_only mode."""
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

        add_safe_globals([TorchVersion, Problem, Resolution, Specifications, ModelCheckpoint, EarlyStopping])
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "data/test_diarization/IyLqUS7hRvo_std_vocals.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_diarization/pyannote_speaker_diarization_3.1"
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return "cpu"
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


def _load_audio_in_memory(audio_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Dict[str, object]:
    waveform, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = waveform.T

    if sr != sample_rate:
        resampled = []
        for channel in waveform:
            resampled.append(scipy.signal.resample_poly(channel, sample_rate, sr).astype(np.float32))
        max_len = max((len(channel) for channel in resampled), default=0)
        waveform = np.stack(
            [np.pad(channel, (0, max_len - len(channel)), mode="constant") for channel in resampled],
            axis=0,
        )

    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    return {
        "waveform": torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
        "sample_rate": sample_rate,
    }


def _load_pyannote_pipeline(model_name: str, token: Optional[str], device: str):
    _register_safe_globals()
    from pyannote.audio import Pipeline

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token = hf_get_token()

    if token:
        try:
            pipeline = Pipeline.from_pretrained(model_name, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
    else:
        pipeline = Pipeline.from_pretrained(model_name)

    if pipeline is None:
        raise RuntimeError(
            f"Failed to load pyannote model: {model_name}. "
            "Make sure you have accepted the model terms on Hugging Face "
            "and are logged in with huggingface-cli login, or pass --hf-token."
        )

    pipeline.to(torch.device(device))
    return pipeline


def _extract_annotation(diarization):
    """Return a pyannote Annotation from either legacy or new pipeline output."""
    return getattr(diarization, "speaker_diarization", diarization)


def _annotation_to_segments(diarization) -> List[Dict]:
    segments = []
    annotation = _extract_annotation(diarization)

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": str(speaker),
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "duration": round(float(turn.end - turn.start), 3),
            }
        )

    segments.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return segments


def _find_overlaps(segments: List[Dict]) -> List[Dict]:
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
                    "start": round(previous_time, 3),
                    "end": round(time, 3),
                    "duration": round(time - previous_time, 3),
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
    pipeline,
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    force: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / f"{input_path.stem}_diarization.json"
    if out_json.exists() and not force:
        return out_json

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    diarization = pipeline(_load_audio_in_memory(input_path), **kwargs)

    segments = _annotation_to_segments(diarization)
    overlaps = _find_overlaps(segments)
    speakers = sorted({seg["speaker"] for seg in segments})

    result = {
        "input": str(input_path.resolve()),
        "model_name": model_name,
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
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    force: bool = False,
) -> List[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    used_device = _resolve_device(device)
    pipeline = _load_pyannote_pipeline(
        model_name=model_name,
        token=hf_token,
        device=used_device,
    )

    input_files = _collect_audio_files(input_path)
    outputs = []
    for in_file in input_files:
        out_json = diarize_single(
            pipeline=pipeline,
            input_path=in_file,
            output_dir=output_dir,
            model_name=model_name,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
            force=force,
        )
        outputs.append(out_json)

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pyannote speaker diarization and output JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input audio file or directory. Default: data/test_diarization/IyLqUS7hRvo_std_vocals.wav",
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
        help="Pyannote diarization model name.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token override. Normally not needed if you already ran `hf auth login`.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers if known.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers if known.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers if known.",
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
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        force=args.force,
    )

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
