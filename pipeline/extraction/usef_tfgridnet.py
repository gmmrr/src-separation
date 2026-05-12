#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USEF-TSE TFGridNet inference pipeline.

This module loads the official USEF-TFGridNet checkpoint and runs inference only.
It does not perform training or fine-tuning.

Default demo inputs:
  - mix: data/raw/test_samples
  - aux: data/test_extraction/IyLqUS7hRvo_std_vocals.wav
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPO_DIR = REPO_ROOT / "models/external_repos/USEF-TSE"
DEFAULT_CHECKPOINT = REPO_ROOT / "models/checkpoints/usef_tse/ZBang_USEF-TSE/chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar"
DEFAULT_MIX = REPO_ROOT / "data/raw/test_samples"
DEFAULT_AUX = REPO_ROOT / "data/test_extraction/IyLqUS7hRvo_std_vocals.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_extraction/usef_tfgridnet"
DEFAULT_SAMPLE_RATE = 8000
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return "cpu"
    return device


def _ensure_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_audio_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _load_audio_mono(audio_path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    return wav


def _select_aux_for_mix(mix_path: Path, aux_inputs: Sequence[Path]) -> Path:
    if len(aux_inputs) == 1:
        return aux_inputs[0]

    stem_matches = [p for p in aux_inputs if p.stem == mix_path.stem]
    if len(stem_matches) == 1:
        return stem_matches[0]

    raise ValueError(
        "Ambiguous auxiliary reference selection. Provide a single aux file or a directory "
        "whose filenames match the mix stems."
    )


def _load_usef_tse_module(repo_dir: Path, model_py: Path):
    repo_dir = repo_dir.resolve()
    model_py = model_py.resolve()

    if not repo_dir.exists():
        raise FileNotFoundError(f"USEF-TSE repo directory not found: {repo_dir}")
    if not model_py.exists():
        raise FileNotFoundError(f"USEF-TFGridNet model file not found: {model_py}")

    _ensure_sys_path(repo_dir)
    module = _load_module_from_file("usef_tfgridnet_model", model_py)
    return module


def _extract_checkpoint_state(ckpt_obj) -> dict:
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(hasattr(v, "shape") for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format: expected a state dict or a wrapper dict.")


def _remap_state_dict_keys(state_dict: dict) -> dict:
    remapped = {}
    for key, value in state_dict.items():
        remapped[key.replace("module.", "")] = value
    return remapped


class STFT(torch.nn.Module):
    def __init__(self, n_fft: int = 256, hop_length: int = 128, win_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y):
        num_dims = y.dim()
        if num_dims not in (2, 3):
            raise ValueError("Only support 2D or 3D input.")

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)

        complex_stft = torch.stft(
            y,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )
        _, num_freqs, _ = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, complex_stft.shape[-1])

        mag = torch.abs(complex_stft)
        phase = torch.angle(complex_stft)
        real = complex_stft.real
        imag = complex_stft.imag
        return mag, phase, real, imag, complex_stft


class iSTFT(torch.nn.Module):
    def __init__(self, n_fft: int = 256, hop_length: int = 128, win_length: int = 256, length=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length

    def forward(self, features, input_type):
        if input_type == "real_imag":
            if not isinstance(features, (tuple, list)):
                raise TypeError("real_imag input expects a tuple or list.")
            real, imag = features
            features = torch.complex(real, imag)
        elif input_type == "complex":
            if not torch.is_complex(features):
                raise TypeError("The input feature is not complex.")
        elif input_type == "mag_phase":
            if not isinstance(features, (tuple, list)):
                raise TypeError("mag_phase input expects a tuple or list.")
            mag, phase = features
            features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        else:
            raise NotImplementedError("Only 'real_imag', 'complex', and 'mag_phase' are supported.")

        return torch.istft(
            features,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=features.device),
            length=self.length,
        )


def build_usef_tfgridnet_model(
    repo_dir: Path | str = DEFAULT_REPO_DIR,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    model_py: Optional[Path | str] = None,
    device: str = "auto",
):
    """Instantiate the USEF-TFGridNet model and load checkpoint weights."""
    repo_dir = Path(repo_dir)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    model_py = Path(model_py) if model_py is not None else checkpoint_dir.parent / "model.py"
    used_device = _resolve_device(device)

    module = _load_usef_tse_module(repo_dir=repo_dir, model_py=model_py)

    # These values match the shipped checkpoint config.
    from models.local.TFgridnet import TF_gridnet_attentionblock

    stft = STFT(n_fft=128, hop_length=64, win_length=128)
    istft = iSTFT(n_fft=128, hop_length=64, win_length=128)
    real_att = TF_gridnet_attentionblock(
        emb_dim=128,
        n_freqs=65,
        n_head=4,
        approx_qk_dim=512,
    )

    model = module.Tar_Model(
        stft=stft,
        istft=istft,
        real_att=real_att,
        n_freqs=65,
        hidden_channels=256,
        n_head=4,
        emb_dim=128,
        emb_ks=1,
        emb_hs=1,
        num_layers=6,
    )

    try:
        ckpt_obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _extract_checkpoint_state(ckpt_obj)
    remapped = _remap_state_dict_keys(state_dict)

    try:
        model.load_state_dict(remapped, strict=True)
    except RuntimeError as exc:
        print("⚠️  Strict checkpoint loading failed, retrying with strict=False.")
        print(f"   {exc}")
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            print(f"   missing keys: {sorted(missing)}")
        if unexpected:
            print(f"   unexpected keys: {sorted(unexpected)}")

    model.to(used_device)
    model.eval()
    return model, used_device


def infer_single(
    model: torch.nn.Module,
    mix_path: Path,
    aux_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    output_stem: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Run USEF-TFGridNet on one mix/aux pair and write the separated waveform."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = output_stem or f"{mix_path.stem}_usef_tfgridnet"
    out_wav = output_dir / f"{out_stem}.wav"
    out_json = output_dir / f"{out_stem}.json"

    if out_wav.exists() and not force:
        return out_wav

    mix = _load_audio_mono(mix_path, sample_rate)
    aux = _load_audio_mono(aux_path, sample_rate)

    mix_len = int(mix.shape[0])
    mix_t = torch.from_numpy(mix).unsqueeze(0).to(device)
    aux_t = torch.from_numpy(aux).unsqueeze(0).to(device)

    with torch.no_grad():
        est = model(mix_t, aux_t)

    est = est.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if est.shape[0] > mix_len:
        est = est[:mix_len]
    elif est.shape[0] < mix_len:
        est = np.pad(est, (0, mix_len - est.shape[0]), mode="constant")

    est = np.clip(est, -1.0, 1.0)
    sf.write(str(out_wav), est, sample_rate, subtype="PCM_16")

    meta = {
        "mix": str(mix_path.resolve()),
        "aux": str(aux_path.resolve()),
        "output": str(out_wav.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "sample_rate": sample_rate,
        "device": device,
        "mix_seconds": round(mix_len / sample_rate, 3),
        "aux_seconds": round(len(aux) / sample_rate, 3),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_wav


def run_usef_tfgridnet(
    mix: Path | str = DEFAULT_MIX,
    aux: Path | str = DEFAULT_AUX,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repo_dir: Path | str = DEFAULT_REPO_DIR,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    model_py: Optional[Path | str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: str = "auto",
    force: bool = False,
) -> List[Path]:
    """Run USEF-TFGridNet inference for a file or directory of mixes."""
    mix_path = Path(mix)
    aux_path = Path(aux)
    output_dir = Path(output_dir)

    model, used_device = build_usef_tfgridnet_model(
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        model_py=model_py,
        device=device,
    )

    mix_files = _collect_audio_files(mix_path)
    aux_inputs = _collect_audio_files(aux_path)

    outputs: List[Path] = []
    for mix_file in mix_files:
        mix_output_dir = output_dir / mix_file.stem

        if len(mix_files) == 1 and len(aux_inputs) > 1:
            for idx, paired_aux in enumerate(aux_inputs, start=1):
                out_path = infer_single(
                    model=model,
                    mix_path=mix_file,
                    aux_path=paired_aux,
                    output_dir=mix_output_dir,
                    checkpoint_path=Path(checkpoint_path),
                    output_stem=f"{idx:02d}_{paired_aux.stem}_usef_tfgridnet",
                    sample_rate=sample_rate,
                    device=used_device,
                    force=force,
                )
                outputs.append(out_path)
            continue

        paired_aux = _select_aux_for_mix(mix_file, aux_inputs)
        out_path = infer_single(
            model=model,
            mix_path=mix_file,
            aux_path=paired_aux,
            output_dir=mix_output_dir,
            checkpoint_path=Path(checkpoint_path),
            output_stem=f"01_{paired_aux.stem}_usef_tfgridnet" if len(aux_inputs) > 1 else f"{mix_file.stem}_usef_tfgridnet",
            sample_rate=sample_rate,
            device=used_device,
            force=force,
        )
        outputs.append(out_path)

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run USEF-TFGridNet inference only.")
    parser.add_argument(
        "--mix",
        type=Path,
        default=DEFAULT_MIX,
        help="Mix audio file or directory. Default: data/raw/test_samples",
    )
    parser.add_argument(
        "--aux",
        type=Path,
        default=DEFAULT_AUX,
        help="Auxiliary target-speaker reference audio file or directory. "
        "Default: data/test_extraction/IyLqUS7hRvo_std_vocals.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where separated audio files will be written.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="Path to the USEF-TSE source repo.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to temp_best.pth.tar for the USEF-TFGridNet checkpoint.",
    )
    parser.add_argument(
        "--model-py",
        type=Path,
        default=None,
        help="Optional path to the model definition file. Default uses the checkpoint-local model.py.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Inference sample rate. USEF-TFGridNet checkpoint expects 8000 Hz.",
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
    outputs = run_usef_tfgridnet(
        mix=args.mix,
        aux=args.aux,
        output_dir=args.output_dir,
        repo_dir=args.repo_dir,
        checkpoint_path=args.checkpoint,
        model_py=args.model_py,
        sample_rate=args.sample_rate,
        device=args.device,
        force=args.force,
    )

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
