from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Tuple, List
from config.constants import KEY_ASR, KEY_VAD
from metadata.utils.io import extract_workspace_audio_ids


# --- Helper: extract unique IDs from Excel ---

def extract_unique_ids(excel_path: Path, id_column="video_id"):
    df = pd.read_excel(excel_path)
    if id_column not in df.columns:
        raise ValueError(
            f"Column '{id_column}' not found in Excel. Columns: {df.columns.tolist()}"
        )
    return sorted(df[id_column].dropna().astype(str).unique())


# --- Helper: union across multiple Excels ---

def extract_unique_ids_union(excel_paths, id_column="video_id"):
    """
    Union IDs across multiple cleaned Excel files (e.g., RA1 + RA2).
    """
    ids = set()
    for p in excel_paths:
        ids.update(extract_unique_ids(Path(p), id_column=id_column))
    return sorted(ids)


# --- evaluable audio-ids ---

def get_evaluable_audio_ids(
    workspace_dir: Path,
    cleaned_excel_paths,
    id_column: str = "video_id",
) -> List[str]:
    """
    Return IDs that:
    - exist as per_audio/<audio_id>/ folders in workspace_dir AND have <audio_id>_metadata.json
    - AND exist in cleaned annotation Excels (id_column contains audio_ids)

    This is the "quiet" variant (no prints).
    """
    pipeline_ids = set(extract_workspace_audio_ids(workspace_dir))
    annotated_ids = set(extract_unique_ids_union(cleaned_excel_paths, id_column=id_column))
    return sorted(pipeline_ids & annotated_ids)


def collect_evaluable_audio_ids(
    workspace_dir: Path,
    cleaned_excel_paths,
    id_column: str = "video_id",
    verbose: bool = True,
) -> List[str]:
    """
    Collect IDs that:
    - exist as per_audio/<audio_id>/ folders in the pipeline workspace (with metadata present)
    - AND exist in the cleaned annotation Excel files

    Returns:
        evaluable_ids (list)

    This is the "verbose" variant (prints counts if verbose=True).
    """
    pipeline_ids = extract_workspace_audio_ids(workspace_dir)
    annotated_ids = extract_unique_ids_union(cleaned_excel_paths, id_column=id_column)
    evaluable_ids = sorted(set(pipeline_ids) & set(annotated_ids))

    if verbose:
        print("Pipeline audio_ids:", len(pipeline_ids))
        print("Annotated audio_ids:", len(annotated_ids))
        print("Evaluable audio_ids (intersection):", len(evaluable_ids))

    return evaluable_ids


def parse_vad_and_asr_identifier_from_audio_id_filename(audio_id: str, filename: str) -> Tuple[str, str]:
    """
    Parse vad_mask and asr_audio_in from filename or filename stem:
        <audio_id>_<vad_mask>_vad_<asr_audio_in>_asr[..._nvv.json]

    IMPORTANT:
    - Raises ValueError if pattern cannot be parsed (desired strict behavior).
    """
    base = filename
    prefix = audio_id + "_"

    if base.startswith(prefix):
        base = base[len(prefix):]

    token_vad = "_vad_"
    token_asr = "_asr"

    if token_vad not in base or token_asr not in base:
        raise ValueError(f"Cannot parse vad/asr sources from filename: {filename}")

    parsed_vad_mask = base.split(token_vad, 1)[0]
    rest = base.split(token_vad, 1)[1]
    parsed_asr_audio_in = rest.split(token_asr, 1)[0]

    return parsed_vad_mask, parsed_asr_audio_in


def derive_combo_key(filename_stem: str, audio_id: str) -> str:
    """
    Derive per-file combo key from filename stem, robust to underscores.
    """
    prefix = f"{audio_id}_"
    return filename_stem[len(prefix):] if filename_stem.startswith(prefix) else filename_stem


def create_combo_key(vad_mask: str, asr_audio_in: str) -> str:
    """
    Build the deterministic combo_key.
    Naming contract: <audio_id>_<vad_mask>_vad_<asr_audio_in>_asr.json
    Therefore: combo_key = <vad_mask>_vad_<asr_audio_in>_asr
    Args:
        vad_mask: VAD mask selector (one of VAD_MASKS, including "no").
        asr_audio_in: ASR input audio derivative (one of AUDIO_DERIVATIVES).
    Returns:
        combo_key string (without audio_id prefix and without file extension).
    """
    return f"{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}"
