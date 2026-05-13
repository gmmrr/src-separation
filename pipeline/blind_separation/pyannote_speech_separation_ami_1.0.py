#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pyannote AMI speech separation pipeline.

This script is a direct inference-only wrapper around the official
``pyannote/speech-separation-ami-1.0`` pipeline, which is built on top of the
``pyannote/separation-ami-1.0`` model.

It writes:
  - a JSON summary with diarization segments and source file paths
  - one WAV file per speaker source
  - an RTTM file for the diarization output

Default demo input:
  - data/test_blind_separation/three_speakers.wav
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torchaudio
from huggingface_hub import get_token as hf_get_token


if not hasattr(torchaudio, "AudioMetaData"):
    class _AudioMetaData:
        """Compatibility shim for pyannote.audio type hints and info return."""

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


def _allow_pyannote_checkpoint_globals() -> None:
    """Allow safe loading of pyannote checkpoints under PyTorch 2.6+."""
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        add_safe_globals([TorchVersion, Problem, Resolution, Specifications])
    except Exception:
        # Older torch versions do not need this, and we do not want loading to fail
        # just because the compatibility hook is unavailable.
        pass


def _patch_speechbrain_embedding_loader() -> None:
    """Patch pyannote's SpeechBrain embedding loader to use SpeechBrain 1.1.x."""
    try:
        import pyannote.audio.pipelines.speaker_verification as sv

        if getattr(sv, "_codex_speechbrain_patch_applied", False):
            return

        def _init(self, embedding="speechbrain/spkrec-ecapa-voxceleb", device=None, use_auth_token=None):
            if not hasattr(sv, "SPEECHBRAIN_IS_AVAILABLE") or not sv.SPEECHBRAIN_IS_AVAILABLE:
                raise ImportError(
                    f"'speechbrain' must be installed to use '{embedding}' embeddings. "
                    "Visit https://speechbrain.github.io for installation instructions."
                )

            self.embedding = embedding.split("@")[0] if isinstance(embedding, str) else embedding
            if isinstance(device, torch.device):
                self.device = str(device)
            elif device is None:
                self.device = "cpu"
            else:
                self.device = str(device)
            self.use_auth_token = use_auth_token

            classifier_cls = sv.SpeechBrain_EncoderClassifier
            self.classifier_ = classifier_cls.from_hparams(
                source=self.embedding,
                savedir=f"{sv.CACHE_DIR}/speechbrain",
                run_opts={"device": self.device},
            )

        def _to(self, device):
            if isinstance(device, torch.device):
                device_str = str(device)
            else:
                device_str = str(device)

            classifier_cls = sv.SpeechBrain_EncoderClassifier
            self.classifier_ = classifier_cls.from_hparams(
                source=self.embedding,
                savedir=f"{sv.CACHE_DIR}/speechbrain",
                run_opts={"device": device_str},
            )
            self.device = device_str
            return self

        sv.SpeechBrainPretrainedSpeakerEmbedding.__init__ = _init
        sv.SpeechBrainPretrainedSpeakerEmbedding.to = _to
        sv._codex_speechbrain_patch_applied = True
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "data/test_blind_separation/three_speakers.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_blind_separation/pyannote_speech_separation_ami_1.0"
DEFAULT_PIPELINE_NAME = "pyannote/speech-separation-ami-1.0"
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
    raise FileNotFoundError(f"Input path does not exist: {path}")


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


def _load_pipeline(pipeline_name: str, token: Optional[str], device: str):
    from pyannote.audio import Pipeline

    _allow_pyannote_checkpoint_globals()
    _patch_speechbrain_embedding_loader()

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token = hf_get_token()

    if token:
        try:
            pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=token)
    else:
        pipeline = Pipeline.from_pretrained(pipeline_name)

    if pipeline is None:
        raise RuntimeError(
            f"Failed to load pyannote pipeline: {pipeline_name}. "
            "Make sure you have accepted the model terms on Hugging Face "
            "and are logged in with huggingface-cli login, or pass --hf-token."
        )

    pipeline.to(torch.device(device))
    return pipeline


def _annotation_to_segments(annotation) -> List[Dict]:
    segments = []
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


def _normalize_for_wav(wav: np.ndarray) -> Tuple[np.ndarray, float, float]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    raw_peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if raw_peak > 0.0:
        wav = wav / raw_peak
    wav = np.clip(wav * 0.99, -1.0, 1.0)
    normalized_peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav, raw_peak, normalized_peak


def _extract_pipeline_outputs(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    raise RuntimeError(
        "Unexpected pyannote separation pipeline output. Expected a tuple "
        "(diarization, sources)."
    )


def _extract_sources_array(sources) -> np.ndarray:
    data = getattr(sources, "data", sources)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def _align_sources_to_speakers(source_array: np.ndarray, num_speakers: int, mix_len: int) -> np.ndarray:
    if source_array.ndim != 2:
        raise RuntimeError(f"Unexpected sources array shape: {tuple(source_array.shape)}")

    if source_array.shape[0] == mix_len:
        return source_array[:, :num_speakers]
    if source_array.shape[1] == mix_len:
        return source_array.T

    if source_array.shape[1] >= num_speakers:
        return source_array[:, :num_speakers]
    if source_array.shape[0] >= num_speakers:
        return source_array[:num_speakers, :].T

    raise RuntimeError(
        f"Could not align sources array shape {tuple(source_array.shape)} "
        f"to {num_speakers} speakers."
    )


def separate_single(
    pipeline,
    input_path: Path,
    output_dir: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / f"{input_path.stem}_separation.json"
    out_rttm = output_dir / f"{input_path.stem}.rttm"
    if out_json.exists() and out_rttm.exists() and not force:
        return out_json

    mix = _load_audio_in_memory(input_path, sample_rate=sample_rate)
    mix_len = int(mix["waveform"].shape[-1])

    if show_progress:
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        with ProgressHook() as hook:
            diarization, sources = _extract_pipeline_outputs(pipeline(mix, hook=hook))
    else:
        diarization, sources = _extract_pipeline_outputs(pipeline(mix))

    annotation = getattr(diarization, "speaker_diarization", diarization)
    segments = _annotation_to_segments(annotation)
    overlaps = _find_overlaps(segments)
    speakers = list(annotation.labels())

    source_array = _extract_sources_array(sources)
    source_array = _align_sources_to_speakers(source_array, len(speakers), mix_len)

    source_paths: List[str] = []
    source_metadata: List[Dict[str, object]] = []
    for idx, speaker in enumerate(speakers):
        if idx >= source_array.shape[1]:
            break

        source = np.asarray(source_array[:, idx], dtype=np.float32)
        if source.shape[0] > mix_len:
            source = source[:mix_len]
        elif source.shape[0] < mix_len:
            source = np.pad(source, (0, mix_len - source.shape[0]), mode="constant")

        source, raw_peak, normalized_peak = _normalize_for_wav(source)

        out_wav = output_dir / f"{idx:02d}_{speaker}.wav"
        sf.write(str(out_wav), source, sample_rate, subtype="PCM_16")
        source_paths.append(str(out_wav.resolve()))
        source_metadata.append(
            {
                "speaker": speaker,
                "output": str(out_wav.resolve()),
                "raw_peak": raw_peak,
                "normalized_peak": normalized_peak,
            }
        )

    diarization_rttm = output_dir / f"{input_path.stem}.rttm"
    with diarization_rttm.open("w", encoding="utf-8") as rttm:
        annotation.write_rttm(rttm)

    result = {
        "input": str(input_path.resolve()),
        "pipeline_name": DEFAULT_PIPELINE_NAME,
        "num_speakers_detected": len(speakers),
        "speakers": speakers,
        "segments": segments,
        "overlaps": overlaps,
        "sample_rate": sample_rate,
        "output_rttm": str(diarization_rttm.resolve()),
        "source_files": source_paths,
        "sources": source_metadata,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_json


def run_separation(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    hf_token: Optional[str] = None,
    device: str = "auto",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force: bool = False,
    show_progress: bool = True,
) -> List[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    used_device = _resolve_device(device)
    pipeline = _load_pipeline(
        pipeline_name=pipeline_name,
        token=hf_token,
        device=used_device,
    )

    input_files = _collect_audio_files(input_path)
    outputs = []
    for in_file in input_files:
        out_json = separate_single(
            pipeline=pipeline,
            input_path=in_file,
            output_dir=output_dir / in_file.stem,
            sample_rate=sample_rate,
            force=force,
            show_progress=show_progress,
        )
        outputs.append(out_json)

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pyannote speech separation and output JSON/WAV files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input audio file or directory. Default: data/test_blind_separation/three_speakers.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where separation outputs will be written.",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default=DEFAULT_PIPELINE_NAME,
        help="Pyannote speech-separation pipeline name.",
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
        help="Target sample rate for the model.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable pyannote ProgressHook output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    outputs = run_separation(
        input_path=args.input,
        output_dir=args.output_dir,
        pipeline_name=args.pipeline_name,
        hf_token=args.hf_token,
        device=args.device,
        sample_rate=args.sample_rate,
        force=args.force,
        show_progress=not args.no_progress,
    )

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
