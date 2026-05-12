#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGMSE speech denoising pipeline.

This script runs the local SGMSE checkpoint in inference-only mode.
It does not train or fine-tune the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SGMSE_REPO_DIR = REPO_ROOT / "models/external_repos/sgmse"
DEFAULT_CHECKPOINT = REPO_ROOT / "models/checkpoints/sgmse/ears_wham.ckpt"
DEFAULT_INPUT = REPO_ROOT / "data/test_denoising"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_denoising/sgmse"
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
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return files

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _ensure_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


@contextmanager
def _temporary_sys_path(path: Path):
    resolved = str(path.resolve())
    inserted = False
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        inserted = True
    try:
        yield
    finally:
        if inserted and sys.path and sys.path[0] == resolved:
            sys.path.pop(0)


def _safe_checkpoint_context():
    _ensure_sys_path(SGMSE_REPO_DIR)
    from sgmse.data_module import Specs, SpecsDataModule

    return torch.serialization.safe_globals([SpecsDataModule, Specs])


def _load_sgmse_model(checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SGMSE checkpoint not found: {checkpoint_path}")
    if not SGMSE_REPO_DIR.exists():
        raise FileNotFoundError(f"SGMSE repo directory not found: {SGMSE_REPO_DIR}")

    with _temporary_sys_path(SGMSE_REPO_DIR):
        from sgmse.model import ScoreModel

        with _safe_checkpoint_context():
            model = ScoreModel.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    model = model.to(device)
    model.eval()
    return model


def _load_audio_mono(audio_path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return wav


def _normalize_waveform(wav: np.ndarray) -> tuple[np.ndarray, float]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.0:
        wav = wav / peak
    return wav, peak


def _chunk_bounds(total_len: int, chunk_len: int, overlap_len: int) -> List[tuple[int, int]]:
    if total_len <= 0:
        return [(0, 0)]
    if chunk_len <= 0 or chunk_len >= total_len:
        return [(0, total_len)]

    overlap_len = max(0, min(overlap_len, chunk_len - 1))
    step = max(1, chunk_len - overlap_len)
    bounds: List[tuple[int, int]] = []
    start = 0
    while start < total_len:
        end = min(start + chunk_len, total_len)
        bounds.append((start, end))
        if end >= total_len:
            break
        start += step
    return bounds


def _blend_weights(length: int, is_first: bool, is_last: bool, overlap_len: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if length <= 1 or overlap_len <= 0:
        return weights

    fade_len = min(overlap_len, max(1, length // 2))
    fade = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)

    if not is_first:
        weights[:fade_len] = fade
    if not is_last:
        weights[-fade_len:] = fade[::-1]
    return weights


def _output_subdir(input_file: Path, input_root: Optional[Path], output_dir: Path) -> Path:
    if input_root is None or input_file.is_absolute() is False:
        return output_dir / input_file.stem

    try:
        rel = input_file.relative_to(input_root)
    except ValueError:
        return output_dir / input_file.stem

    if rel.suffix:
        rel = rel.with_suffix("")
    return output_dir / rel


def _enhance_waveform(
    model,
    waveform: np.ndarray,
    device: str,
    N: int,
    snr: float,
    corrector_steps: int,
    sampler_type: str,
    corrector: str,
    t_eps: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> np.ndarray:
    _ensure_sys_path(SGMSE_REPO_DIR)
    from sgmse.util.other import pad_spec

    chunk_len = int(round(chunk_seconds * 16000))
    overlap_len = int(round(overlap_seconds * 16000))
    bounds = _chunk_bounds(len(waveform), chunk_len, overlap_len)

    def _enhance_chunk(chunk: np.ndarray) -> np.ndarray:
        y = torch.from_numpy(chunk).unsqueeze(0).to(device)
        norm_factor = y.abs().max().clamp(min=1e-12)
        y = y / norm_factor

        T_orig = y.size(1)
        Y = torch.unsqueeze(model._forward_transform(model._stft(y)), 0)
        Y = pad_spec(Y)

        model.t_eps = t_eps

        if model.sde.__class__.__name__ == "OUVESDE":
            if sampler_type == "pc":
                sampler = model.get_pc_sampler(
                    "reverse_diffusion",
                    corrector,
                    Y.to(device),
                    N=N,
                    corrector_steps=corrector_steps,
                    snr=snr,
                    intermediate=False,
                )
            elif sampler_type == "ode":
                sampler = model.get_ode_sampler(Y.to(device), N=N)
            else:
                raise ValueError(f"Unsupported sampler type for OUVESDE: {sampler_type}")
        elif model.sde.__class__.__name__ == "SBVESDE":
            local_sampler_type = "ode" if sampler_type == "pc" else sampler_type
            sampler = model.get_sb_sampler(sde=model.sde, y=Y.to(device), sampler_type=local_sampler_type)
        else:
            raise ValueError(f"Unsupported SDE type: {model.sde.__class__.__name__}")

        with torch.no_grad():
            sample, _ = sampler()
            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

        return x_hat.squeeze().detach().cpu().numpy().astype(np.float32)

    if len(bounds) == 1:
        return _enhance_chunk(waveform)

    print(
        f"Chunking single file into {len(bounds)} parts "
        f"({chunk_seconds:.1f}s chunks, {overlap_seconds:.1f}s overlap)."
    )
    accum = np.zeros(len(waveform), dtype=np.float32)
    weights = np.zeros(len(waveform), dtype=np.float32)

    for idx, (start, end) in enumerate(tqdm(bounds, desc="SGMSE chunks", unit="chunk"), start=1):
        chunk_start = time.perf_counter()
        is_first = start == 0
        is_last = end >= len(waveform)
        chunk = waveform[start:end]
        chunk_weights = _blend_weights(len(chunk), is_first, is_last, overlap_len)

        print(f"  chunk {idx}/{len(bounds)}: samples {start}:{end}", flush=True)
        enhanced = _enhance_chunk(chunk)

        if enhanced.shape[0] != len(chunk):
            if enhanced.shape[0] > len(chunk):
                enhanced = enhanced[: len(chunk)]
            else:
                enhanced = np.pad(enhanced, (0, len(chunk) - enhanced.shape[0]), mode="constant")

        accum[start:end] += enhanced * chunk_weights
        weights[start:end] += chunk_weights
        elapsed = time.perf_counter() - chunk_start
        print(f"  chunk {idx}/{len(bounds)} done in {elapsed:.1f}s", flush=True)

    weights = np.maximum(weights, 1e-8)
    return accum / weights


def _write_output(
    out_wav: Path,
    out_json: Path,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model,
    device: str,
    sample_rate: int,
    norm_peak: float,
    enhanced: np.ndarray,
    input_seconds: float,
    N: int,
    snr: float,
    corrector_steps: int,
    sampler_type: str,
    corrector: str,
    t_eps: float,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), enhanced, sample_rate, subtype="PCM_16")

    meta = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": device,
        "sample_rate": sample_rate,
        "model_backbone": getattr(model, "backbone", None),
        "model_sde": model.sde.__class__.__name__,
        "model_sr": getattr(model, "sr", None),
        "n_fft": model.data_module.n_fft,
        "hop_length": model.data_module.hop_length,
        "num_frames": model.data_module.num_frames,
        "spec_factor": model.data_module.spec_factor,
        "spec_abs_exponent": model.data_module.spec_abs_exponent,
        "normalize": model.data_module.normalize,
        "sampler_type": sampler_type,
        "corrector": corrector,
        "corrector_steps": corrector_steps,
        "snr": snr,
        "N": N,
        "t_eps": t_eps,
        "input_seconds": round(input_seconds, 3),
        "input_peak": norm_peak,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def infer_single(
    model,
    input_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    device: str,
    sample_rate: int,
    N: int,
    snr: float,
    corrector_steps: int,
    sampler_type: str,
    corrector: str,
    t_eps: float,
    chunk_seconds: float,
    overlap_seconds: float,
    force: bool = False,
    input_root: Optional[Path] = None,
) -> Path:
    out_dir = _output_subdir(input_path, input_root, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_wav = out_dir / f"{input_path.stem}_sgmse.wav"
    out_json = out_dir / f"{input_path.stem}_sgmse.json"

    if out_wav.exists() and out_json.exists() and not force:
        return out_wav

    waveform = _load_audio_mono(input_path, sample_rate=sample_rate)
    input_seconds = len(waveform) / float(sample_rate) if sample_rate > 0 else 0.0
    enhanced = _enhance_waveform(
        model=model,
        waveform=waveform,
        device=device,
        N=N,
        snr=snr,
        corrector_steps=corrector_steps,
        sampler_type=sampler_type,
        corrector=corrector,
        t_eps=t_eps,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    if enhanced.shape[0] > waveform.shape[0]:
        enhanced = enhanced[: waveform.shape[0]]
    elif enhanced.shape[0] < waveform.shape[0]:
        enhanced = np.pad(enhanced, (0, waveform.shape[0] - enhanced.shape[0]), mode="constant")

    enhanced, peak = _normalize_waveform(enhanced)
    enhanced = np.clip(enhanced * 0.99, -1.0, 1.0)

    _write_output(
        out_wav=out_wav,
        out_json=out_json,
        input_path=input_path,
        output_path=out_wav,
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        sample_rate=sample_rate,
        norm_peak=peak,
        enhanced=enhanced,
        input_seconds=input_seconds,
        N=N,
        snr=snr,
        corrector_steps=corrector_steps,
        sampler_type=sampler_type,
        corrector=corrector,
        t_eps=t_eps,
    )
    return out_wav


def run_sgmse(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    N: int = 30,
    snr: float = 0.5,
    corrector_steps: int = 1,
    sampler_type: str = "pc",
    corrector: str = "ald",
    t_eps: float = 0.03,
    chunk_seconds: float = 8.0,
    overlap_seconds: float = 1.0,
    force: bool = False,
) -> List[Path]:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    used_device = _resolve_device(device)
    model = _load_sgmse_model(checkpoint_path=checkpoint_path, device=used_device)
    sample_rate = int(getattr(model, "sr", 16000) or 16000)

    input_files = _collect_audio_files(input_path)
    outputs: List[Path] = []
    for in_file in tqdm(input_files, desc="SGMSE files", unit="file"):
        start = time.perf_counter()
        out_path = infer_single(
            model=model,
            input_path=in_file,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            device=used_device,
            sample_rate=sample_rate,
            N=N,
            snr=snr,
            corrector_steps=corrector_steps,
            sampler_type=sampler_type,
            corrector=corrector,
            t_eps=t_eps,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            force=force,
            input_root=input_path if input_path.is_dir() else None,
        )
        outputs.append(out_path)
        elapsed = time.perf_counter() - start
        print(f"Processed {in_file.name} in {elapsed:.1f}s")

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SGMSE denoising inference.")
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
        help="Directory where denoised outputs will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the local SGMSE checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=30,
        help="Number of reverse diffusion steps.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.5,
        help="SNR value for the Langevin corrector.",
    )
    parser.add_argument(
        "--corrector-steps",
        type=int,
        default=1,
        help="Number of corrector steps.",
    )
    parser.add_argument(
        "--sampler-type",
        type=str,
        choices=("pc", "ode"),
        default="pc",
        help="Sampler type.",
    )
    parser.add_argument(
        "--corrector",
        type=str,
        choices=("ald", "langevin", "none"),
        default="ald",
        help="Corrector used by the PC sampler.",
    )
    parser.add_argument(
        "--t-eps",
        type=float,
        default=0.03,
        help="Minimum process time.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=8.0,
        help="Chunk length for single-file progress reporting. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        help="Overlap between chunks when chunking is enabled.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    outputs = run_sgmse(
        input_path=args.input,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        N=args.N,
        snr=args.snr,
        corrector_steps=args.corrector_steps,
        sampler_type=args.sampler_type,
        corrector=args.corrector,
        t_eps=args.t_eps,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        force=args.force,
    )

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
