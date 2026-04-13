#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STEP 2: SOURCE SEPARATION (UVR-MDX-Net Inst3) ---
# toDo: Update source separation model to better handle noisy audio (e.g., reverberation).

import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from metadata.utils.io import ensure_dir, read_json, write_json
from pipeline.pipeline_workspace_runner import setup_workspace_run
from metadata.metadata import audio_dir_metadata_path, set_metadata_audio, mark_step
from metadata.utils.detect_device import detect_device
from pipeline.separate_fast import Predictor
from config.constants import (
    AUDIO_STD,
    KEY_STEP_2,
    EXT_WAV,
    KEY_VOCALS,
    KEY_BACKGROUND,
    KEY_AUDIO_FILES,
    KEY_FIELD_PATH,
    KEY_FIELD_SR,
    KEY_FIELD_CHANNELS,
    AUDIO_STD_VOCALS,
    AUDIO_STD_BACKGROUND,
)
from config.params import (
    STEP2_TARGET_SR,
    STEP2_MODEL_N_FFT,
    STEP2_MODEL_DIM_F,
    STEP2_MODEL_DIM_T,
    STEP2_MODEL_CHUNKS,
    STEP2_MODEL_MARGIN,
    STEP2_MODEL_DENOISE,
    STEP2_WAV_SUBTYPE,
)

# --- Predictor Loader (helper) ---

def load_step_2_predictor(
    model_path: Path | str,
    device: str = "auto",
) -> Predictor:
    """
    Create and return a UVR / MDX-Net Predictor instance.
    This is a reusable helper so you can load the model once (e.g., in batch runners)
    and reuse it across many files.
    Args:
        model_path: Path to the ONNX model. (UVR-MDX-Net Inst3 (ONNX))
        device: "auto" | "cuda" | "cpu"
    Returns:
        Predictor instance (ready to .predict()).
    """
    used_device = detect_device(device) if device == "auto" else device
    model_path = Path(model_path).resolve()

    predictor = Predictor(
        args={
            "model_path": str(model_path),
            "dim_f": STEP2_MODEL_DIM_F,
            "dim_t": STEP2_MODEL_DIM_T,
            "n_fft": STEP2_MODEL_N_FFT,
            "chunks": STEP2_MODEL_CHUNKS,
            "margin": STEP2_MODEL_MARGIN,
            "denoise": STEP2_MODEL_DENOISE,
        },
        device=used_device,
    )
    return predictor


# --- Single File Processing ---

def source_separate_single_audio(
    audio_id_dir: Path,
    predictor: Predictor,
    device: str = "auto",
    force: bool = False
):
    """
    Step 2: Source Separation using UVR-MDX-Net Inst3
    Output: separated vocals and background files, keeping full suffix chain
    Args:
      - audio_id_dir: Input: standardized file from Step 1 (audios/std)
      - predictor: Pre-loaded Predictor instance
      - device: Only for logging (already resolved if using load_step_2_predictor)
      - force: Whether to overwrite existing outputs

    Error philosophy:
      - If required input is missing although Step 1 created metadata, this should raise.

    """
    # --- Setup ---
    t0 = time.time()

    # Resolve device only if needed (standalone-friendly)
    used_device = detect_device(device) if device == "auto" else device

    meta_path = audio_dir_metadata_path(audio_id_dir)
    meta = read_json(meta_path)

    # --- Skip if already processed ---
    if meta.get("steps", {}).get(KEY_STEP_2, {}).get("status") == "done" and not force:
        print(f"↪ {audio_id_dir.name}: Step 2 already done (cached)")
        return meta

    # --- Resolve input file ---
    in_path_str = meta.get(KEY_AUDIO_FILES, {}).get(AUDIO_STD, {}).get(KEY_FIELD_PATH, "")
    in_path = Path(in_path_str) if in_path_str else None

    if in_path is None or not in_path.exists():
        raise FileNotFoundError(
            f"❌ Step 2 input not found. Expected metadata['{KEY_AUDIO_FILES}']['{AUDIO_STD}']['{KEY_FIELD_PATH}'] "
            f"to point to an existing file, got: {in_path_str}"
        )

    print(f"▶ Step 2: Source Separation for {audio_id_dir.name}, input: {in_path.name}")

    # --- Load input audio ---
    mix, _ = librosa.load(in_path, sr=STEP2_TARGET_SR, mono=False)
    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=0)

    # --- Perform separation ---
    background, target = predictor.predict(mix)
    # NOTE: Swapping background and target to match expected output 
    # "vocals" are usually interpreted as "background singers"
    vocals, background = background.astype(np.float32), target.astype(np.float32)

    # Ensure correct shape and clipping
    def _ensure_shape(a: np.ndarray) -> np.ndarray:
        return a.T if a.shape[0] == 2 else a

    vocals, background = map(lambda x: np.clip(_ensure_shape(x), -1, 1), [vocals, background])

    # --- Save output files ---
    audio_dir = audio_id_dir / KEY_AUDIO_FILES
    ensure_dir(audio_dir)

    # preserve full suffix chain (e.g. "_std" → "_std_vocals")
    stem = in_path.stem
    out_voc = audio_dir / f"{stem}_{KEY_VOCALS}{EXT_WAV}"
    out_bg = audio_dir / f"{stem}_{KEY_BACKGROUND}{EXT_WAV}"

    sf.write(out_voc, vocals, STEP2_TARGET_SR, subtype=STEP2_WAV_SUBTYPE)
    sf.write(out_bg, background, STEP2_TARGET_SR, subtype=STEP2_WAV_SUBTYPE)

    # --- Update metadata ---
    set_metadata_audio(meta, AUDIO_STD_VOCALS, out_voc, STEP2_TARGET_SR, 2)
    set_metadata_audio(meta, AUDIO_STD_BACKGROUND, out_bg, STEP2_TARGET_SR, 2)

    # --- Log Step ---
    mark_step(
        meta,
        KEY_STEP_2,
        "done",
        t0,
        {
            KEY_FIELD_SR: STEP2_TARGET_SR,
            KEY_FIELD_CHANNELS: 2,
            "device": used_device,
            "input": str(in_path),
            # optional traceability for tuning (can be removed later if too noisy)
            "uvr_params": {
                "dim_f": STEP2_MODEL_DIM_F,
                "dim_t": STEP2_MODEL_DIM_T,
                "n_fft": STEP2_MODEL_N_FFT,
                "chunks": STEP2_MODEL_CHUNKS,
                "margin": STEP2_MODEL_MARGIN,
                "denoise": STEP2_MODEL_DENOISE,
            },
        },
    )

    # --- Write metadata ---
    write_json(meta_path, meta)
    print(f"✓ Separation done → {out_voc.name}, {out_bg.name}")
    return meta


# --- Batch / Workspace Runner ---

def run_step_2_separate(
    workspace: Path | str,
    model_path: Path | str,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Batch runner for Step 2.
    Runs only Step 2 over all per_audio/<audio_id>/ folders that have metadata.
    Resolves device once per workspace run.
    """
    setup = setup_workspace_run(
        workspace=workspace,
        device=device,
        force=force,
        input_dir=None,
        require_metadata=True,
    )

    used_device = setup["device"]
    audio_id_dirs = setup["audio_id_dirs"]
    if not audio_id_dirs:
        print("⚠️  No per_audio folders with metadata found. Run Step 1 first.")
        return
    
    predictor = load_step_2_predictor(model_path=model_path, device=used_device)    

    print(f"🚀 Step 2 (Separation) for workspace='{Path(workspace).name}'")
    print(f"   • Model:  {model_path}")
    print(f"   • Device: {used_device}")
    print(f"   • Clips:  {len(audio_id_dirs)}")

    for audio_id_dir in audio_id_dirs:
        source_separate_single_audio(
            audio_id_dir=audio_id_dir,
            predictor=predictor,
            device=used_device, 
            force=setup["force"],
        )

    print("✅ Step 2 completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 2 (source separation) over a workspace.")
    parser.add_argument("--workspace", required=True, help="Output workspace root directory")
    parser.add_argument("--model", required=True, help="Path to UVR-MDX-Net model (.onnx)")
    parser.add_argument("--device", default="auto", help="Device to use: auto | cuda | cpu")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_step_2_separate(
        workspace=args.workspace,
        model_path=args.model,
        device=args.device,
        force=args.force,
    )
