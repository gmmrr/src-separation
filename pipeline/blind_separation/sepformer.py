#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SepFormer blind separation pipeline.

This module runs SpeechBrain SepFormer inference only, using the pretrained
speechbrain/sepformer-wsj02mix model. It does not perform training.

Default demo input:
  - data/test_blind_separation/IyLqUS7hRvo_std_vocals.wav
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]

SEPFORMER_SOURCE = "speechbrain/sepformer-wsj02mix"
SEPFORMER_SAVEDIR = REPO_ROOT / "models/checkpoints/speechbrain_sepformer_wsj02mix"

DEFAULT_INPUT = REPO_ROOT / "data/test_blind_separation/IyLqUS7hRvo_std_vocals.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_blind_separation/sepformer_outputs"
DEFAULT_SAMPLE_RATE = 8000
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
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _load_audio_mono(audio_path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.to(torch.float32)
    if wav.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)


def _load_sepformer(source: str, savedir: Path, device: str):
    from speechbrain.inference.separation import SepformerSeparation

    savedir.mkdir(parents=True, exist_ok=True)
    return SepformerSeparation.from_hparams(
        source=source,
        savedir=str(savedir),
        run_opts={"device": device},
    )


def _normalize_for_wav(wav: np.ndarray) -> tuple[np.ndarray, float, float]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    raw_peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if raw_peak > 0.0:
        wav = wav / raw_peak
    wav = np.clip(wav * 0.99, -1.0, 1.0)
    normalized_peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav, raw_peak, normalized_peak


def infer_single(
    separator,
    input_path: Path,
    output_dir: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force: bool = False,
) -> List[Path]:
    """Run SepFormer on one input file and write the separated waveforms."""
    output_dir.mkdir(parents=True, exist_ok=True)

    out_original = output_dir / "00_original.wav"
    out_prefix = "sepformer"

    if force or not out_original.exists():
        shutil.copy2(input_path, out_original)
    mix = _load_audio_mono(input_path, sample_rate)
    mix_len = int(mix.shape[0])

    with torch.no_grad():
        separated = separator.separate_batch(mix.unsqueeze(0).to(separator.device))
    separated = separated.detach().cpu()

    if separated.ndim == 3:
        separated = separated.squeeze(0)
    if separated.ndim != 2:
        raise RuntimeError(f"SepFormer inference returned unexpected shape: {tuple(separated.shape)}")

    if separated.shape[0] == mix_len:
        separated = separated.transpose(0, 1)
    if separated.shape[1] != mix_len:
        raise RuntimeError(f"SepFormer output length mismatch: {tuple(separated.shape)}, input length: {mix_len}")

    output_paths: List[Path] = []
    for idx, enhanced_stream in enumerate(separated, start=1):
        out_wav = output_dir / f"{idx:02d}_{out_prefix}.wav"
        out_json = output_dir / f"{idx:02d}_{out_prefix}.json"
        if out_wav.exists() and out_json.exists() and not force:
            output_paths.append(out_wav)
            continue

        enhanced = enhanced_stream.numpy().astype(np.float32)
        if enhanced.shape[0] > mix_len:
            enhanced = enhanced[:mix_len]
        elif enhanced.shape[0] < mix_len:
            enhanced = np.pad(enhanced, (0, mix_len - enhanced.shape[0]), mode="constant")

        enhanced, raw_peak, normalized_peak = _normalize_for_wav(enhanced)
        sf.write(str(out_wav), enhanced, sample_rate, subtype="PCM_16")

        meta = {
            "input": str(input_path.resolve()),
            "output": str(out_wav.resolve()),
            "original_copy": str(out_original.resolve()),
            "model_source": SEPFORMER_SOURCE,
            "model_savedir": str(SEPFORMER_SAVEDIR.resolve()),
            "sample_rate": sample_rate,
            "device": str(separator.device),
            "required_channels": 1,
            "input_seconds": round(mix_len / sample_rate, 3),
            "raw_peak": raw_peak,
            "normalized_peak": normalized_peak,
            "stream_index": idx,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        output_paths.append(out_wav)

    return output_paths


def run_sepformer(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_source: str = SEPFORMER_SOURCE,
    model_savedir: Path | str = SEPFORMER_SAVEDIR,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: str = "auto",
    force: bool = False,
) -> List[Path]:
    """Run SepFormer blind separation for a file or directory of files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    model_savedir = Path(model_savedir)

    used_device = _resolve_device(device)
    separator = _load_sepformer(
        source=model_source,
        savedir=model_savedir,
        device=used_device,
    )

    input_files = _collect_audio_files(input_path)
    outputs: List[Path] = []
    for in_file in input_files:
        out_dir = output_dir / in_file.stem
        out_paths = infer_single(
            separator=separator,
            input_path=in_file,
            output_dir=out_dir,
            sample_rate=sample_rate,
            force=force,
        )
        outputs.extend(out_paths)

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SepFormer blind separation.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input audio file or directory. Default: data/test_blind_separation/IyLqUS7hRvo_std_vocals.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where enhanced outputs will be written.",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default=SEPFORMER_SOURCE,
        help="SpeechBrain SepFormer model source. Default: speechbrain/sepformer-wsj02mix.",
    )
    parser.add_argument(
        "--model-savedir",
        type=Path,
        default=SEPFORMER_SAVEDIR,
        help="Directory where the SpeechBrain model is cached.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate used for blind separation. Default: 8000.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs if they already exist.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    outputs = run_sepformer(
        input_path=args.input,
        output_dir=args.output_dir,
        model_source=args.model_source,
        model_savedir=args.model_savedir,
        sample_rate=args.sample_rate,
        device=args.device,
        force=args.force,
    )
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
