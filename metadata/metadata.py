from pathlib import Path
import time
from typing import Any, Dict, Optional
from config.constants import KEY_ASR, KEY_VAD, KEY_METADATA, KEY_AUDIO_FILES, KEY_FIELD_PATH, KEY_FIELD_SR, KEY_FIELD_CHANNELS, KEY_STEP_LOG, KEY_ANNOTATIONS, KEY_LABELS
from metadata.utils.io import read_json, read_json_with_status, write_json, audio_dir_metadata_path


def set_metadata_audio(meta: dict, key: str, path: Path, sr: int, channels: int):
    """Convenience helper to update audio metadata."""
    meta.setdefault(KEY_AUDIO_FILES, {})
    meta[KEY_AUDIO_FILES][key] = {
        "path": str(path),
        "sr": int(sr),
        "channels": int(channels)
    }


def update_metadata(metadata_path: Path, annotation_key: str, annotation_path: str):
    """
    Append or update an annotation entry in metadata.json.
    Creates structure:
    {
      "annotations": {
         "<annotation_key>": {
             "path": "...",
             "updated": "<timestamp>"
         }
      }
    }
    """
    meta = read_json(metadata_path) if metadata_path.exists() else {}

    meta.setdefault(KEY_ANNOTATIONS, {})
    meta[KEY_ANNOTATIONS].setdefault(annotation_key, {})

    meta[KEY_ANNOTATIONS][annotation_key].update({
        KEY_FIELD_PATH: annotation_path,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    write_json(metadata_path, meta)
    print(f"📄 Metadata updated: {metadata_path.name} → {annotation_key}")

                  
#urgent toDo: renaming required for updated single source of truth of audio_derivatives and vad_masks instead of sources and source_types!
def update_metadata_with_label(work_dir: Path, label_path: Path, source: str, audio_derivative: str, generated_from: Path):
    """
    Update metadata.json with info about a generated label file.

    Args:
        work_dir (Path): Directory containing metadata.json
        label_path (Path): Path to generated label file
        source (str): e.g. 'vad', 'asr', etc.
        source_type (str): e.g. 'vocals_norm', 'original'
        generated_from (Path): JSON file that the label was created from
    """
    from metadata.utils.io import read_json, write_json
    meta_path = audio_dir_metadata_path(work_dir)
    if not meta_path.exists():
        print(f"⚠️  Metadata file not found: {meta_path}")
        return
    meta = read_json(meta_path)
    meta.setdefault(KEY_LABELS, {})
    meta[KEY_LABELS].setdefault(source, {})  # ✅ create nested group, e.g. "labels"["vad"]

    label_entry = {
        "path": str(label_path.resolve()),
        "source": source,
        "source_type": audio_derivative,
        "format": "audacity",
        "generated_from": Path(generated_from).name,
        "time_s": 0.0,
    }

    meta[KEY_LABELS][source][label_path.name] = label_entry  # ✅ store inside labels["vad"][filename]

    write_json(meta_path, meta)


def reset_metadata_group(
    meta: Dict[str, Any],
    parent_key: str,
    group_key: str,
) -> int:
    """
    Reset (delete and recreate) a nested metadata group. (Preventing mixing old and new entries)
    Args:
        meta: The metadata dict to modify in-place.
        parent_key: Top-level key (e.g., KEY_ANNOTATIONS).
        group_key: Nested group key (e.g., KEY_ASR).
    Returns:
        Number of keys removed from the previous group (0 if group didn't exist).
    """
    meta.setdefault(parent_key, {})
    prev = meta[parent_key].get(group_key)

    removed = 0
    if isinstance(prev, dict):
        removed = len(prev)

    meta[parent_key][group_key] = {}
    return removed


def mark_step(
    meta: Dict[str, Any],
    step_name: str,
    status: str = "done",
    t0: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update step log in metadata.

    Args:
        meta: The metadata dict.
        step_name: Pipeline step key (e.g., KEY_STEP_5).
        status: Status string ("done", "error", ...).
        t0: Start time for runtime measurement.
        extra: Additional step metadata fields.
    """
    meta.setdefault(KEY_STEP_LOG, {})
    meta[KEY_STEP_LOG].setdefault(step_name, {})
    step_meta = meta[KEY_STEP_LOG][step_name]

    step_meta["status"] = status
    if t0 is not None:
        step_meta["time_s"] = round(time.time() - t0, 3)
    if extra:
        step_meta.update(extra)
    
