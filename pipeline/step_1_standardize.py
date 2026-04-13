#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STEP 1: STANDARDIZATION (Input for UVR) ---

import time
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from metadata.utils.io import ensure_dir, read_json, write_json
from metadata.metadata import audio_dir_metadata_path, set_metadata_audio, mark_step
from config.constants import (
    KEY_AUDIO_ID,
    KEY_STEP_1,
    EXT_WAV,
    KEY_ORIGINAL,
    KEY_STD,
    AUDIO_STD,
    KEY_AUDIO_FILES,
    KEY_FIELD_SR,
    KEY_FIELD_CHANNELS,
)
from config.params import STEP2_TARGET_SR

# --- Single File Processing ---

def standardize_single_audio(
    input_file: Path,
    audio_id_dir: Path,
    device: str = "auto",
    force: bool = False,
):
    """
    Step 1: Standardization (Input Preparation for Source Separation)
      - Stereo, SEPARATION_SAMPLING_RATE Hz
      - Peak normalization [-1, 1]
      - No RMS leveling (peak normalization only)
    """

    # --- Setup ---
    t0 = time.time()
    meta_path = audio_dir_metadata_path(audio_id_dir)
    meta = read_json(meta_path)
    audio_id = audio_id_dir.name

    # --- Skip if already processed ---
    if meta.get("steps", {}).get(KEY_STEP_1, {}).get("status") == "done" and not force:
        print(f"↪ {audio_id_dir.name}: Step 1 already done (cached)")
        return meta

    # --- Ensure audio output folder ---
    audio_dir = audio_id_dir / KEY_AUDIO_FILES
    ensure_dir(audio_dir)

    # --- Load input audio ---
    y, sr_in = librosa.load(input_file, sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
        orig_channels = 1
    else:
        orig_channels = int(y.shape[0])

    # --- Store original file info in metadata ---
    orig_sr = int(sr_in)
    meta[KEY_AUDIO_ID] = audio_id
    set_metadata_audio(meta, KEY_ORIGINAL, input_file.resolve(), orig_sr, orig_channels)

    # --- Resample to target sampling rate ---
    if sr_in != STEP2_TARGET_SR:
        y = np.stack(
            [
                librosa.resample(y=y[0], orig_sr=sr_in, target_sr=STEP2_TARGET_SR),
                librosa.resample(y=y[1], orig_sr=sr_in, target_sr=STEP2_TARGET_SR),
            ],
            axis=0,
        )
        sr_out = STEP2_TARGET_SR
    else:
        sr_out = int(sr_in)

    # --- Peak normalization ---
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = y / peak

    # --- Save standardized file ---
    out_path = audio_dir / f"{audio_id}_{KEY_STD}{EXT_WAV}"
    sf.write(out_path, y.T, sr_out, subtype="PCM_16")

    # --- Update metadata ---
    set_metadata_audio(meta, AUDIO_STD, out_path, sr_out, 2)
    mark_step(
        meta,
        KEY_STEP_1,
        "done",
        t0,
        {
            KEY_FIELD_SR: int(sr_out),
            KEY_FIELD_CHANNELS: 2,
            "device": device,
        },
    )

    # --- Write metadata ---
    write_json(meta_path, meta)
    print(f"Step 1: ✓ Saved standardized file → {out_path.name}")
    return meta



# --- Batch / Workspace Runner ---
def run_step_1_std(
    input_dir: Path | str,
    workspace: Path | str,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Run Step 1 (standardization) for all audio files in the given input directory.
    For each supported audio file a subfolder under `workspace/per_audio/<audio_id>` is
    created, the file is standardized and metadata are updated.

    Args:
        input_dir (Path|str): Directory containing raw audio files.
        workspace (Path|str): Root directory for storing per_audio data.
        device (str, optional): Device flag forwarded to step_1_standardize ("auto", "cuda", "cpu").
        force (bool, optional): Overwrite existing standardization results if True.
    """
    from pipeline.pipeline_workspace_runner import setup_workspace_run

    setup = setup_workspace_run(
        workspace=workspace,
        input_dir=input_dir,
        device=device,
        force=force,
        require_metadata=False,  # Step 1 first run may have no metadata yet
    )

    input_files = setup["input_files"]
    per_audio_dir = setup["per_audio_dir"]
    used_device = setup["device"]

    for file in input_files:
        audio_id = file.stem
        audio_id_dir = per_audio_dir / audio_id
        ensure_dir(audio_id_dir)

        standardize_single_audio(file, audio_id_dir, device=used_device, force=force)

    print("✅ Step 1 Normalization completed.") 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Step 1 (standardization) for all audio files in a directory."
    )
    parser.add_argument("--input", required=True, help="Input directory with raw audio files")
    parser.add_argument("--workspace", required=True, help="Output workspace root directory")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use for processing: auto | cuda | cpu",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing standardization outputs",
    )
    args = parser.parse_args()
    run_step_1_std(args.input, args.workspace, device=args.device, force=args.force)
