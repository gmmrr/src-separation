#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metadata_evaluation.py
---------------------------------------------------------
Minimal helpers to store evaluation information in per-track metadata.json.

Design goals:
- Keep it minimal and overwrite-safe (no endless run history).
- Store one "latest" evaluation run under meta["evaluation"]["run"].
- Store per-track results under meta["evaluation"]["tracks"][track_key].
---------------------------------------------------------
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from metadata.utils.io import read_json, write_json, read_json_with_status
from metadata.metadata import audio_dir_metadata_path


def _now_ts() -> str:
    """Return a human-readable timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _make_run_id(prefix: str = "eval") -> str:
    """Create a simple run id that changes every run (overwrite-safe)."""
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def mark_evaluation_run(
    work_dir: Path,
    vocals_xlsx: Path,
    params: Dict[str, Any],
    outputs: Dict[str, Any],
) -> Optional[str]:
    """
    Initialize/overwrite evaluation metadata for a single track directory (per_audio/<audio_id>/).

    Writes:
      meta["evaluation"]["run"] = {...}
      meta["evaluation"]["tracks"] = {}  (reset)

    Args:
        work_dir: per_audio/<audio_id>/ directory (NOT the workspace root).
        vocals_xlsx: Path to the cleaned GT excel used.
        params: Evaluation parameters (e.g., min_overlap_ms, skip_existing).
        outputs: Output paths (csv/xlsx) or other run outputs.

    Returns:
        run_id (str) if metadata exists and was updated, else None.
    """
    work_dir = Path(work_dir)
    meta_file = audio_dir_metadata_path(work_dir)
    if not meta_file.exists():
        # Keep evaluation runnable even if metadata is missing
        print(f"⚠️  Metadata file not found (skip metadata eval): {meta_file}")
        return None

    meta = read_json(meta_file)
    run_id = _make_run_id()

    meta["evaluation"] = {
        "run": {
            "run_id": run_id,
            "timestamp": _now_ts(),
            "gt_excel": str(Path(vocals_xlsx).resolve()),
            "params": params,
            "outputs": outputs,
        },
        "tracks": {},
    }

    write_json(meta_file, meta)
    return run_id


def update_evaluation_track(
    work_dir: Path,
    track_key: str,
    eval_status: str,
    counts: Dict[str, int],
    paths: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
) -> bool:
    """
    Update evaluation results for one candidate track (one *_nvv.json file).

    Args:
        work_dir: per_audio/<audio_id>/ directory.
        track_key: Unique key for the evaluated track (recommendation: candidate filename).
        eval_status: "ok" | "valid_empty" | "error" | "skipped" (minimal set).
        counts: Dict with counts (gt/cand/hit/miss/bonus).
        paths: Optional dict of relevant paths (e.g., candidate_file, eval_xlsx).
        note: Optional short note.

    Returns:
        True if metadata was updated, else False.
    """
    work_dir = Path(work_dir)
    meta_file = audio_dir_metadata_path(work_dir)
    if not meta_file.exists():
        return False

    meta = read_json(meta_file)
    meta.setdefault("evaluation", {})
    meta["evaluation"].setdefault("tracks", {})

    entry = {
        "eval_status": eval_status,
        "counts": {k: int(v) for k, v in (counts or {}).items()},
        "updated": _now_ts(),
    }
    if paths:
        entry["paths"] = paths
    if note:
        entry["note"] = note

    meta["evaluation"]["tracks"][track_key] = entry
    write_json(meta_file, meta)
    return True


def finalize_evaluation_run(
    work_dir: Path,
    summary: Dict[str, Any],
) -> bool:
    """
    Add/overwrite a mini summary for the evaluation run.

    Args:
        work_dir: per_audio/<audio_id>/ directory.
        summary: Small run summary (e.g., num_tracks, totals, notes).

    Returns:
        True if metadata was updated, else False.
    """
    work_dir = Path(work_dir)
    meta_file = audio_dir_metadata_path(work_dir)
    if not meta_file.exists():
        return False

    meta = read_json(meta_file)
    if "evaluation" not in meta or "run" not in meta["evaluation"]:
        return False

    meta["evaluation"]["run"]["summary"] = summary
    meta["evaluation"]["run"]["updated"] = _now_ts()
    write_json(meta_file, meta)
    return True


def count_nvv_candidates(candidate_path: Path) -> int:
    """
    Count candidates in a Step-7 NVV JSON file.

    Returns:
      - >= 0 : valid JSON + dict + key "nvv" is a list => len(list)
      - -1   : missing/broken/unexpected structure/unreadable
    """
    data, status = read_json_with_status(candidate_path)
    if status != "ok":
        return -1
    if not isinstance(data, dict):
        return -1
    nvv = data.get("nvv")
    if not isinstance(nvv, list):
        return -1
    return len(nvv)


def is_nvv_json_empty(candidate_path: Path) -> bool:
    """
    Backwards-compatible helper: True iff valid JSON and {"nvv": []}.
    """
    return count_nvv_candidates(candidate_path) == 0
