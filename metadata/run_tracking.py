"""Lightweight deterministic run tracking per workspace.

Generates a resume-safe run identity (SHA-256 hash), writes run.json to each
workspace, and maintains an append-only runs_index.json at the processed_root
for grid/screening traceability.

Hash inputs: dataset name, input relative path, pipeline step configs
             (including all tunable YAML params for Steps 4, 5, 7),
             and remaining pipeline params from params.py.
NOT hashed:  timestamps, git hash, absolute machine paths, evaluation params, output paths.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config.constants import KEY_PER_AUDIO
from config.path_factory import get_run_json_path, get_runs_index_json_path
from metadata.utils.io import read_json, write_json

# ── file names kept for backward compatibility ─────────────────────────────────────────────────────
_RUN_JSON = "run.json"
_RUNS_INDEX_JSON = "runs_index.json"


# ── internal snapshots ────────────────────────────────────────────────────────

def _pipeline_params() -> dict[str, Any]:
    """Pipeline-relevant params (STEP1-7) from params.py – excluded: EVAL_*."""
    import config.params as p
    return {
        k: v for k, v in vars(p).items()
        if not k.startswith("_") and not k.startswith("EVAL_")
    }


def _all_params() -> dict[str, Any]:
    """All non-private params from params.py (used for storage only)."""
    import config.params as p
    return {k: v for k, v in vars(p).items() if not k.startswith("_")}


def _pipeline_config_dict(config: Any) -> dict[str, Any]:
    """Serialize pipeline step settings to a plain dict."""
    return {
        "step_4_vad": {
            "vad_audios_in": config.step_4_vad.vad_audios_in,
            "vad_threshold": config.step_4_vad.vad_threshold,
            "vad_min_speech_ms": config.step_4_vad.vad_min_speech_ms,
            "vad_min_silence_ms": config.step_4_vad.vad_min_silence_ms,
            "vad_pad_ms": config.step_4_vad.vad_pad_ms,
        },
        "step_5_asr": {
            "vad_masks_in": config.step_5_asr.vad_masks_in,
            "asr_audios_in": config.step_5_asr.asr_audios_in,
            "asr_chunk_length_s": config.step_5_asr.asr_chunk_length_s,
            "asr_batch_size": config.step_5_asr.asr_batch_size,
        },
        "step_6_nlp": {"spacy_model": config.step_6_nlp.spacy_model},
        "step_7_nvv": {
            "exclude_categories": config.step_7_nvv.exclude_categories,
            "min_duration": config.step_7_nvv.min_duration,
            "max_duration": config.step_7_nvv.max_duration,
            "vad_masks_in": config.step_7_nvv.vad_masks_in,
            "asr_audios_in": config.step_7_nvv.asr_audios_in,
            "vad_gate_padding": config.step_7_nvv.vad_gate_padding,
            "dedup_overlap_ratio": config.step_7_nvv.dedup_overlap_ratio,
            "dedup_time_tol_s": config.step_7_nvv.dedup_time_tol_s,
        },
    }


def _yaml_snapshot(cfg_path: Path) -> dict[str, Any]:
    """Raw YAML config as dict (stored in run.json for reference only)."""
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}


def _rel(path: Path, root: Path) -> str:
    """Return path relative to root, or the absolute string if not under root."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ── public API ────────────────────────────────────────────────────────────────

def compute_run_hash(config: Any, dataset: Any) -> str:
    """Return a 16-char deterministic SHA-256 hex digest for a workspace run.

    The hash captures everything that determines pipeline output:
    dataset identity, pipeline step config, and tunable params (STEP1-7).
    It is stable across machines and time (no timestamps, no git hash).
    """
    payload = {
        "dataset_name": dataset.name,
        "input_rel": _rel(dataset.input_dir, config.project.raw_root),
        "pipeline": _pipeline_config_dict(config),
        "params": _pipeline_params(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_run_json(workspace: Path) -> dict[str, Any]:
    """Return the contents of <workspace>/run.json, or {} if absent.

    Delegates to read_json() which returns {} for missing files.
    """
    return read_json(get_run_json_path(workspace))


def write_run(config: Any, dataset: Any, *, force: bool = False) -> dict[str, Any]:
    """Write (or verify) run.json for a workspace and update the run index.

    - Fresh workspace: creates run.json and appends to runs_index.json.
    - Resume (same hash): run.json is refreshed, no new index entry added.
    - Config changed (hash mismatch): raises RuntimeError.
    - run.json absent but artifacts exist: raises RuntimeError unless force=True.

    Args:
        config: Loaded PipelineConfig.
        dataset: DatasetPaths for the workspace.
        force: If True, allow overwriting a stale workspace (existing artifacts
               without run.json). Must match the pipeline's force flag so the
               caller can also reprocess all artifacts.

    Returns the run data dict that was written.
    """
    workspace = Path(dataset.workspace)
    run_id = compute_run_hash(config, dataset)

    existing = load_run_json(workspace)
    if existing:
        if existing.get("run_id") != run_id:
            raise RuntimeError(
                f"Run hash mismatch in workspace: {workspace}\n"
                f"  stored  run_id: {existing.get('run_id')}\n"
                f"  current run_id: {run_id}\n"
                "Config changed since last run. "
                "Use a different workspace (change output_rel in config.yaml), "
                "or set force=True to reprocess all artifacts with the new config."
            )
    else:
        # run.json missing — check for stale artifacts
        per_audio = workspace / KEY_PER_AUDIO
        try:
            has_artifacts = per_audio.exists() and next(per_audio.iterdir(), None) is not None
        except OSError:
            has_artifacts = False
        if has_artifacts and not force:
            raise RuntimeError(
                f"Workspace has existing artifacts but no run.json: {workspace}\n"
                "Cannot verify config compatibility with existing artifacts.\n"
                "Use a different workspace (change output_rel in config.yaml), "
                "or set force=True to reprocess all artifacts with the new config."
            )

    now = datetime.now(timezone.utc).isoformat()
    output_rel = _rel(dataset.workspace, config.project.processed_root)
    input_rel = _rel(dataset.input_dir, config.project.raw_root)

    run_data: dict[str, Any] = {
        "run_id": run_id,
        "dataset_name": dataset.name,
        "input_rel": input_rel,
        "output_rel": output_rel,
        "created_at": now,
        "pipeline_config": _pipeline_config_dict(config),
        "params": _all_params(),
        "config_yaml": _yaml_snapshot(config.cfg_path),
    }

    write_json(get_run_json_path(workspace), run_data)
    _append_runs_index(config.project.processed_root, run_id, dataset.name, output_rel, now)

    return run_data


# ── runs index ────────────────────────────────────────────────────────────────

def _append_runs_index(
    processed_root: Path,
    run_id: str,
    dataset_name: str,
    output_rel: str,
    created_at: str,
) -> None:
    """Append an entry to <processed_root>/runs_index.json (append-only).

    Duplicate (run_id, output_rel) pairs are ignored. (idempotent / resume-safe).
    """
    index_path = get_runs_index_json_path(processed_root)
    index: list[dict[str, Any]] = []

    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            index = []

    # Note: read-check-write is not atomic; acceptable for single-process use.
    if any(e.get("run_id") == run_id and e.get("output_rel") == output_rel for e in index):
        return  # already recorded; nothing to do

    index.append({
        "run_id": run_id,
        "dataset_name": dataset_name,
        "output_rel": output_rel,
        "created_at": created_at,
    })

    write_json(index_path, index)
