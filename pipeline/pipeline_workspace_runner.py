#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from metadata.utils.io import ensure_dir
from metadata.utils.detect_device import detect_device
from metadata.metadata import audio_dir_metadata_path
from config.constants import KEY_PER_AUDIO


def setup_workspace_run(
    workspace: Union[str, Path],
    device: str = "auto",
    force: bool = False,
    input_dir: Optional[Union[str, Path]] = None,
    supported_exts: Sequence[str] = (".wav"),
    require_metadata: bool = True,
) -> Dict[str, object]:
    """
    Minimal shared setup for workspace-based pipeline steps.

    Use cases:
      - Step 1: pass input_dir to discover raw files and create per_audio/<audio_id>/ folders.
      - Steps 2–7: omit input_dir and iterate existing per_audio/<audio_id>/ folders.

    Args:
        workspace: Workspace root path.
        device: "auto" | "cuda" | "cpu"
        force: Forwarded force flag.
        input_dir: Optional raw input directory. If provided, returns input_files.
        supported_exts: Flat input discovery extensions.
        require_metadata: If True, audio_id_dirs will include only folders with metadata present.

    Returns:
        Dict with keys:
          - workspace (Path)
          - per_audio_dir (Path)
          - device (str): resolved device
          - force (bool)
          - input_dir (Path|None)
          - input_files (List[Path])
          - audio_id_dirs (List[Path])
    """
    workspace = Path(workspace).resolve()
    ensure_dir(workspace)

    per_audio_dir = workspace / KEY_PER_AUDIO
    ensure_dir(per_audio_dir)

    resolved_device = detect_device(device)

    # Optional raw input discovery
    in_dir = Path(input_dir).resolve() if input_dir is not None else None
    input_files: List[Path] = []
    if in_dir is not None:
        input_files = [
            p for p in in_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        ]

    # per_audio discovery
    audio_id_dirs: List[Path] = []
    for p in per_audio_dir.iterdir():
        if not p.is_dir():
            continue
        if require_metadata:
            if not audio_dir_metadata_path(p).exists():
                continue
        audio_id_dirs.append(p)

    return {
        "workspace": workspace,
        "per_audio_dir": per_audio_dir,
        "device": resolved_device,
        "force": force,
        "input_dir": in_dir,
        "input_files": input_files,
        "audio_id_dirs": audio_id_dirs,
    }
