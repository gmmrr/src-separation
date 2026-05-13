#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TFGridNet blind separation pipeline.

This module runs ESPnet2 TFGridNet inference only, using the local checkpoint
under models/checkpoints/tfgridnet/. It does not perform training.

Default demo input:
  - data/test_blind_separation/IyLqUS7hRvo_std_vocals.wav
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

ESPnet_REPO_DIR = REPO_ROOT / "models/external_repos/espnet"
TFGRIDNET_DIR = REPO_ROOT / "models/checkpoints/tfgridnet/yoshiki_wsj0_2mix_spatialized_enh_tfgridnet_waspaa2023_raw"
TFGRIDNET_TRAIN_CONFIG = TFGRIDNET_DIR / "exp/enh_train_enh_tfgridnet_waspaa2023_raw/config.yaml"
TFGRIDNET_MODEL_FILE = TFGRIDNET_DIR / "exp/enh_train_enh_tfgridnet_waspaa2023_raw/25epoch.pth"

DEFAULT_INPUT = REPO_ROOT / "data/test_blind_separation/IyLqUS7hRvo_std_vocals.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_blind_separation/tfgridnet"
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


def _ensure_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_audio_multi(audio_path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]
    elif wav.ndim == 2:
        wav = wav.T
    else:
        raise ValueError(f"Unsupported audio shape: {wav.shape}")
    return wav


def _match_channels(wav: np.ndarray, target_channels: int) -> np.ndarray:
    if wav.ndim != 2:
        raise ValueError(f"Expected 2D waveform [T, C], got {wav.shape}")
    current_channels = wav.shape[1]
    if current_channels == target_channels:
        return wav
    if current_channels > target_channels:
        return wav[:, :target_channels]
    reps = (target_channels + current_channels - 1) // current_channels
    wav = np.tile(wav, (1, reps))
    return wav[:, :target_channels]


def _load_separate_speech(
    espnet_repo_dir: Path,
    train_config: Path,
    model_file: Path,
    device: str,
):
    if not espnet_repo_dir.exists():
        raise FileNotFoundError(f"ESPnet repo directory not found: {espnet_repo_dir}")
    if not train_config.exists():
        raise FileNotFoundError(f"TFGridNet train_config not found: {train_config}")
    if not model_file.exists():
        raise FileNotFoundError(f"TFGridNet checkpoint not found: {model_file}")

    _ensure_sys_path(espnet_repo_dir)
    from espnet2.bin.enh_inference import SeparateSpeech

    return SeparateSpeech(
        train_config=str(train_config),
        model_file=str(model_file),
        normalize_output_wav=False,
        device=device,
        dtype="float32",
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
    separate_speech,
    input_path: Path,
    output_dir: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force: bool = False,
) -> List[Path]:
    """Run TFGridNet on one input file and write the enhanced waveform."""
    output_dir.mkdir(parents=True, exist_ok=True)

    out_original = output_dir / "00_original.wav"
    out_prefix = "tfgridnet"

    if force or not out_original.exists():
        shutil.copy2(input_path, out_original)
    mix = _load_audio_multi(input_path, sample_rate)
    required_channels = getattr(getattr(separate_speech.enh_model, "separator", None), "n_imics", mix.shape[1])
    mix = _match_channels(mix, int(required_channels))
    mix_len = int(mix.shape[0])

    enhanced_list = separate_speech(mix[None, :, :], fs=sample_rate)
    if not enhanced_list:
        raise RuntimeError("TFGridNet inference returned no output streams.")

    output_paths: List[Path] = []
    for idx, enhanced_stream in enumerate(enhanced_list, start=1):
        out_wav = output_dir / f"{idx:02d}_{out_prefix}.wav"
        out_json = output_dir / f"{idx:02d}_{out_prefix}.json"
        if out_wav.exists() and out_json.exists() and not force:
            output_paths.append(out_wav)
            continue

        enhanced = np.asarray(enhanced_stream[0], dtype=np.float32)
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
            "train_config": str(TFGRIDNET_TRAIN_CONFIG.resolve()),
            "model_file": str(TFGRIDNET_MODEL_FILE.resolve()),
            "sample_rate": sample_rate,
            "device": getattr(separate_speech, "device", "unknown"),
            "required_channels": int(required_channels),
            "input_seconds": round(mix_len / sample_rate, 3),
            "raw_peak": raw_peak,
            "normalized_peak": normalized_peak,
            "stream_index": idx,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        output_paths.append(out_wav)

    return output_paths


def run_tfgridnet(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    espnet_repo_dir: Path | str = ESPnet_REPO_DIR,
    train_config: Path | str = TFGRIDNET_TRAIN_CONFIG,
    model_file: Path | str = TFGRIDNET_MODEL_FILE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: str = "auto",
    force: bool = False,
) -> List[Path]:
    """Run TFGridNet blind separation for a file or directory of files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    espnet_repo_dir = Path(espnet_repo_dir)
    train_config = Path(train_config)
    model_file = Path(model_file)

    used_device = _resolve_device(device)
    separate_speech = _load_separate_speech(
        espnet_repo_dir=espnet_repo_dir,
        train_config=train_config,
        model_file=model_file,
        device=used_device,
    )

    input_files = _collect_audio_files(input_path)
    outputs: List[Path] = []
    for in_file in input_files:
        out_dir = output_dir / in_file.stem
        out_paths = infer_single(
            separate_speech=separate_speech,
            input_path=in_file,
            output_dir=out_dir,
            sample_rate=sample_rate,
            force=force,
        )
        outputs.extend(out_paths)

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TFGridNet blind separation.")
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
        "--espnet-repo",
        type=Path,
        default=ESPnet_REPO_DIR,
        help="Path to the local ESPnet repository.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=TFGRIDNET_TRAIN_CONFIG,
        help="Path to TFGridNet config.yaml from the checkpoint folder.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=TFGRIDNET_MODEL_FILE,
        help="Path to TFGridNet .pth checkpoint file.",
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
    outputs = run_tfgridnet(
        input_path=args.input,
        output_dir=args.output_dir,
        espnet_repo_dir=args.espnet_repo,
        train_config=args.train_config,
        model_file=args.model_file,
        sample_rate=args.sample_rate,
        device=args.device,
        force=args.force,
    )
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
