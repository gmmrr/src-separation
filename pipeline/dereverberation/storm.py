#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StoRM dereverberation pipeline.

This script runs the local StoRM checkpoint in inference-only mode.
It does not train or fine-tune the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import scipy.signal
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
STORM_REPO_DIR = REPO_ROOT / "models/external_repos/storm"
DEFAULT_CHECKPOINT = REPO_ROOT / "models/checkpoints/storm/WSJ0+Reverb/epoch=237-pesq=0.00.ckpt"
DEFAULT_INPUT = REPO_ROOT / "data/test_dereverberation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_dereverberation/storm"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
MODEL_SAMPLE_RATE = 16000


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _collect_audio_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return files

    raise FileNotFoundError(f"Input path does not exist: {path}")


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


@contextmanager
def _cpu_safe_storm_imports():
    """Import StoRM modules without requiring CUDA extensions on CPU-only hosts."""
    import torch.utils.cpp_extension as cpp_extension

    should_stub = not torch.cuda.is_available() or torch.version.cuda is None
    original_load = cpp_extension.load
    dummy_ext = types.SimpleNamespace(
        fused_bias_act=lambda *args, **kwargs: None,
        upfirdn2d=lambda *args, **kwargs: None,
    )

    if should_stub:
        cpp_extension.load = lambda *args, **kwargs: dummy_ext

    try:
        with _temporary_sys_path(STORM_REPO_DIR):
            from sgmse.data_module import Specs, SpecsDataModule
            from sgmse.model import StochasticRegenerationModel
            from sgmse.util.other import pad_spec

        yield StochasticRegenerationModel, SpecsDataModule, Specs, pad_spec
    finally:
        if should_stub:
            cpp_extension.load = original_load


def _load_storm_model(checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"StoRM checkpoint not found: {checkpoint_path}")
    if not STORM_REPO_DIR.exists():
        raise FileNotFoundError(f"StoRM repo directory not found: {STORM_REPO_DIR}")

    with _cpu_safe_storm_imports() as (StormModel, SpecsDataModule, Specs, pad_spec):
        with torch.serialization.safe_globals([SpecsDataModule, Specs]):
            model = StormModel.load_from_checkpoint(
                str(checkpoint_path),
                base_dir="",
                batch_size=1,
                num_workers=0,
                kwargs=dict(gpu=False),
                map_location="cpu",
            )

    model.eval(no_ema=False)
    model.to(device)
    return model, pad_spec


def _load_audio_mono(audio_path: Path, sample_rate: int) -> np.ndarray:
    waveform, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = waveform.mean(axis=1)

    if sr != sample_rate:
        waveform = scipy.signal.resample_poly(waveform, sample_rate, sr).astype(np.float32)

    return np.asarray(waveform, dtype=np.float32).reshape(-1)


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


def _output_path(input_file: Path, input_root: Optional[Path], output_dir: Path) -> Path:
    if input_root is not None:
        try:
            rel = input_file.relative_to(input_root)
        except ValueError:
            rel = input_file.name
        else:
            rel = rel.with_suffix("")
        if isinstance(rel, Path):
            return output_dir / rel.parent / f"{rel.name}_storm.wav"
        return output_dir / f"{input_file.stem}_storm.wav"

    return output_dir / f"{input_file.stem}_storm.wav"


def _enhance_waveform(
    model,
    pad_spec,
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
    chunk_len = int(round(chunk_seconds * MODEL_SAMPLE_RATE))
    overlap_len = int(round(overlap_seconds * MODEL_SAMPLE_RATE))
    bounds = _chunk_bounds(len(waveform), chunk_len, overlap_len)

    if len(bounds) == 1:
        y = torch.from_numpy(waveform).unsqueeze(0).to(device)
        norm_factor = y.abs().max().clamp(min=1e-12)
        y = y / norm_factor
        T_orig = y.size(1)

        Y = torch.unsqueeze(model._forward_transform(model._stft(y)), 0)
        Y = pad_spec(Y)
        model.t_eps = t_eps

        with torch.no_grad():
            if getattr(model, "denoiser_net", None) is not None:
                Y_denoised = model.forward_denoiser(Y)
            else:
                Y_denoised = None

            if getattr(model, "score_net", None) is not None:
                if model.condition == "noisy":
                    score_conditioning = [Y]
                elif model.condition == "post_denoiser":
                    score_conditioning = [Y_denoised]
                elif model.condition == "both":
                    score_conditioning = [Y, Y_denoised]
                else:
                    raise NotImplementedError(
                        f"Unsupported StoRM conditioning mode: {model.condition}"
                    )

                if sampler_type == "pc":
                    sampler = model.get_pc_sampler(
                        "reverse_diffusion",
                        corrector,
                        Y_denoised,
                        N=N,
                        corrector_steps=corrector_steps,
                        snr=snr,
                        intermediate=False,
                        conditioning=score_conditioning,
                    )
                elif sampler_type == "ode":
                    sampler = model.get_ode_sampler(
                        Y_denoised,
                        N=N,
                        conditioning=score_conditioning,
                    )
                else:
                    raise ValueError(f"Unsupported sampler type: {sampler_type}")

                sample, _ = sampler()
            else:
                sample = Y_denoised

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

        return x_hat.squeeze().detach().cpu().numpy().astype(np.float32)

    print(
        f"Processing in {len(bounds)} chunks "
        f"({chunk_seconds:.1f}s chunks, {overlap_seconds:.1f}s overlap)."
    )
    accum = np.zeros(len(waveform), dtype=np.float32)
    weights = np.zeros(len(waveform), dtype=np.float32)

    for idx, (start, end) in enumerate(tqdm(bounds, desc="StoRM chunks", unit="chunk"), start=1):
        chunk_start = time.perf_counter()
        is_first = start == 0
        is_last = end >= len(waveform)
        chunk = waveform[start:end]
        chunk_weights = _blend_weights(len(chunk), is_first, is_last, overlap_len)

        print(f"  chunk {idx}/{len(bounds)}: start samples {start}:{end}", flush=True)

        y = torch.from_numpy(chunk).unsqueeze(0).to(device)
        norm_factor = y.abs().max().clamp(min=1e-12)
        y = y / norm_factor
        T_orig = y.size(1)

        Y = torch.unsqueeze(model._forward_transform(model._stft(y)), 0)
        Y = pad_spec(Y)
        model.t_eps = t_eps

        with torch.no_grad():
            if getattr(model, "denoiser_net", None) is not None:
                Y_denoised = model.forward_denoiser(Y)
            else:
                Y_denoised = None

            if getattr(model, "score_net", None) is not None:
                if model.condition == "noisy":
                    score_conditioning = [Y]
                elif model.condition == "post_denoiser":
                    score_conditioning = [Y_denoised]
                elif model.condition == "both":
                    score_conditioning = [Y, Y_denoised]
                else:
                    raise NotImplementedError(
                        f"Unsupported StoRM conditioning mode: {model.condition}"
                    )

                if sampler_type == "pc":
                    sampler = model.get_pc_sampler(
                        "reverse_diffusion",
                        corrector,
                        Y_denoised,
                        N=N,
                        corrector_steps=corrector_steps,
                        snr=snr,
                        intermediate=False,
                        conditioning=score_conditioning,
                    )
                elif sampler_type == "ode":
                    sampler = model.get_ode_sampler(
                        Y_denoised,
                        N=N,
                        conditioning=score_conditioning,
                    )
                else:
                    raise ValueError(f"Unsupported sampler type: {sampler_type}")

                sample, _ = sampler()
            else:
                sample = Y_denoised

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

        enhanced = x_hat.squeeze().detach().cpu().numpy().astype(np.float32)
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
    checkpoint_path: Path,
    model,
    device: str,
    sample_rate: int,
    input_seconds: float,
    input_peak: float,
    output_audio: np.ndarray,
    N: int,
    snr: float,
    corrector_steps: int,
    sampler_type: str,
    corrector: str,
    t_eps: float,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), output_audio, sample_rate, subtype="FLOAT")

    meta = {
        "input": str(input_path.resolve()),
        "output": str(out_wav.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": device,
        "sample_rate": sample_rate,
        "model_backbone": getattr(model, "backbone", None),
        "model_sde": model.sde.__class__.__name__,
        "model_condition": getattr(model, "condition", None),
        "model_sr": getattr(model, "sr", None),
        "n_fft": model.data_module.n_fft,
        "hop_length": model.data_module.hop_length,
        "num_frames": model.data_module.num_frames,
        "spec_factor": model.data_module.spec_factor,
        "spec_abs_exponent": model.data_module.spec_abs_exponent,
        "sampler_type": sampler_type,
        "corrector": corrector,
        "corrector_steps": corrector_steps,
        "snr": snr,
        "N": N,
        "t_eps": t_eps,
        "input_seconds": round(input_seconds, 3),
        "input_peak": input_peak,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def infer_single(
    model,
    pad_spec,
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
    out_wav = _output_path(input_path, input_root, output_dir)
    out_json = out_wav.with_suffix(".json")

    if out_wav.exists() and out_json.exists() and not force:
        return out_wav

    waveform = _load_audio_mono(input_path, sample_rate=sample_rate)
    input_seconds = len(waveform) / float(sample_rate) if sample_rate > 0 else 0.0
    input_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0

    enhanced = _enhance_waveform(
        model=model,
        pad_spec=pad_spec,
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

    _write_output(
        out_wav=out_wav,
        out_json=out_json,
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        sample_rate=sample_rate,
        input_seconds=input_seconds,
        input_peak=input_peak,
        output_audio=enhanced,
        N=N,
        snr=snr,
        corrector_steps=corrector_steps,
        sampler_type=sampler_type,
        corrector=corrector,
        t_eps=t_eps,
    )
    return out_wav


def run_dereverberation(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    device: str = "auto",
    N: int = 50,
    snr: float = 0.5,
    corrector_steps: int = 1,
    sampler_type: str = "pc",
    corrector: str = "ald",
    t_eps: float = 0.03,
    chunk_seconds: float = 6.0,
    overlap_seconds: float = 1.0,
    force: bool = False,
) -> List[Path]:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    used_device = _resolve_device(device)
    model, pad_spec = _load_storm_model(checkpoint_path=checkpoint_path, device=used_device)
    sample_rate = int(getattr(model, "sr", MODEL_SAMPLE_RATE) or MODEL_SAMPLE_RATE)

    input_files = _collect_audio_files(input_path)
    outputs: List[Path] = []
    for in_file in input_files:
        out_path = infer_single(
            model=model,
            pad_spec=pad_spec,
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

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StoRM dereverberation inference.")
    parser.add_argument(
        "--input",
        "--input-dir",
        "--input_dir",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input audio file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where enhanced outputs will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the local StoRM checkpoint.",
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
        default=50,
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
        default=6.0,
        help="Chunk length for long files. Use 0 to disable chunking.",
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

    outputs = run_dereverberation(
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

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
