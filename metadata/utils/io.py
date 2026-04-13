from __future__ import annotations
from pathlib import Path
import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, List

import yaml
from config.constants import KEY_METADATA, KEY_PER_AUDIO


def ensure_dir(path: Path) -> None:
    """
    Creates the given directory recursively if it does not exist.
    Args: path (Path): path that should exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)


def audio_dir_metadata_path(audio_id_dir: Path) -> Path:
    """
    Metadata file is always:
        per_audio/<audio_id>/<audio_id>_metadata.json
    """
    audio_id = audio_id_dir.name
    return audio_id_dir / f"{audio_id}_{KEY_METADATA}.json"


def read_json(path: Path) -> Dict[str, Any]:
    """
    Safely read a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: Loaded JSON data (empty dict if file is missing or invalid).
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"\u26a0\ufe0f  Error reading {path.name}: {e}")
        return {}


def read_json_with_status(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Read JSON and return a status string to distinguish missing vs broken files.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        (data, status)
        - status="ok":      file exists and JSON was parsed successfully
        - status="missing": file does not exist
        - status="error":   file exists but cannot be parsed/read
    """
    if not path.exists():
        return None, "missing"

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "ok"
    except (json.JSONDecodeError, OSError) as e:
        print(f"\u26a0\ufe0f  Error reading {path.name}: {e}")
        return None, "error"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Writes a JSON file safely using UTF-8 encoding.

    This function writes to a temporary file first and then replaces the target file
    atomically. This prevents corrupted/partial JSON files if the process is interrupted
    during writing.

    Args:
        path (Path): Target path.
        data (dict): Data to serialize.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = None
    tmp_path = None

    try:
        # Create temp file in the same directory to allow atomic replace
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        tmp_path = Path(tmp_name)

        # Write JSON to temp file
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is physically written to disk

        fd = None  # fd is owned/closed by os.fdopen context manager

        # Atomically replace target file
        os.replace(str(tmp_path), str(path))

    except OSError as e:
        print(f"\u274c Error writing JSON to {path.name}: {e}")

    finally:
        # Cleanup: if something failed before replace, remove temp file
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass

        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def is_audio_id_dir(audio_id_dir: Path) -> bool:
    """
    A valid per_audio/<audio_id>/ folder is defined by:
    - directory exists
    - and per your convention: metadata file exists at <audio_id>_metadata.json
    """
    return audio_id_dir.is_dir() and audio_dir_metadata_path(audio_id_dir).exists()


def extract_workspace_audio_ids(workspace_dir: Path) -> List[str]:
    """
    Extract unique audio IDs from a workspace.

    Required structure (strict):
        <workspace>/
          per_audio/
            <audio_id>/
              <audio_id>_metadata.json

    Returns:
        Sorted list of audio_id folder names that are "valid" by is_audio_id_dir().

    Raises:
        FileNotFoundError if workspace or per_audio folder does not exist.
    """
    workspace_dir = Path(workspace_dir)
    if not workspace_dir.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace_dir}")

    per_audio_dir = workspace_dir / KEY_PER_AUDIO
    if not per_audio_dir.exists():
        raise FileNotFoundError(f"per_audio directory not found: {per_audio_dir}")

    audio_ids: List[str] = []
    for p in per_audio_dir.iterdir():
        if is_audio_id_dir(p):
            audio_ids.append(p.name)

    return sorted(audio_ids)

def print_header(title: str, subtitle: Optional[str] = None) -> None:
    print("\n" + "=" * 80)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 80)

def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return a dict.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
