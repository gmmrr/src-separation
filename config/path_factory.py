from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List
import yaml

from config.constants import (
    KEY_PER_AUDIO,
    KEY_GLOBAL,
    KEY_CLIPS,
    KEY_EVALUATION,
    KEY_AUDIO_FILES,
    KEY_ANNOTATIONS,
    KEY_LABELS,
    KEY_METADATA,
    KEY_VAD,
    KEY_ASR,
    KEY_NLP,
    KEY_NVV,
    KEY_STD,
    KEY_VOCALS,
    KEY_BACKGROUND,
    KEY_NORM,
    EXT_WAV,
    EXT_JSON,
    EXT_TXT,
    KEY_RQ,
)

# --- Run tracking file names ---
_RUN_JSON = "run.json"
_RUNS_INDEX_JSON = "runs_index.json"

# --- Evaluation xlsx base name prefix ---
_GLOBAL_EVALUATION_XLSX_BASE = "_global_evaluation"
_PIPELINE_CAPABILITY_XLSX_BASE = "_pipeline_capability"


# --- Dataclasses ---

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    raw_root: Path
    processed_root: Path


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    input_dir: Path
    workspace: Path


@dataclass(frozen=True)
class WorkspacePaths:
    workspace: Path
    per_audio: Path
    global_dir: Path
    clips: Path
    evaluation: Path


@dataclass(frozen=True)
class GtExcelUnitPaths:
    """
    One GT unit as defined in YAML under evaluation.gt_units.
    """

    name: str

    # RAW
    raw_excel_path: Path
    vocals_dataset_root: Optional[Path]

    # PROCESSED
    cleaned_excel_path: Path
    labels_export_dir: Path

    # Optional VOCALS subset copy (RAW)
    vocals_subset_copy_dir: Optional[Path]

    # Meta
    id_column: str
    gt_mode: str


@dataclass(frozen=True)
class EvaluationDatasetPaths:
    name: str
    workspace: Path
    results_dir: Path


@dataclass(frozen=True)
class EvaluationPaths:
    gt_units: List[GtExcelUnitPaths]
    dataset_results: List[EvaluationDatasetPaths]
    truth_excels: List[Path]


# --- YAML Loader ---

def _load_yaml(cfg_path: Path) -> dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Project + Workspace ---

def get_project_paths(cfg_path: str | Path) -> ProjectPaths:
    """
    Resolve project root, raw root and processed root.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg_dir = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    project_root = (cfg_dir / cfg["project"]["root"]).resolve()
    raw_root = (project_root / cfg["paths"]["raw_root"]).resolve()
    processed_root = (project_root / cfg["paths"]["processed_root"]).resolve()

    return ProjectPaths(
        project_root=project_root,
        raw_root=raw_root,
        processed_root=processed_root,
    )


def get_datasets(
    cfg_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> List[DatasetPaths]:
    """
    Resolve pipeline datasets defined under workspace.datasets.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)
    project = project or get_project_paths(cfg_path)

    ws_cfg = cfg.get("workspace", {})
    datasets_cfg = ws_cfg.get("datasets", [])

    if not datasets_cfg:
        raise ValueError("workspace.datasets is empty or missing in YAML.")

    datasets: List[DatasetPaths] = []

    for ds in datasets_cfg:
        name = str(ds["name"])
        input_rel = str(ds["input_rel"])
        output_rel = str(ds["output_rel"])

        input_dir = (project.raw_root / input_rel).resolve()
        workspace = (project.processed_root / output_rel).resolve()

        datasets.append(
            DatasetPaths(
                name=name,
                input_dir=input_dir,
                workspace=workspace,
            )
        )

    return datasets


def get_workspace_paths(workspace: Path) -> WorkspacePaths:
    """
    Build standard workspace structure.
    """
    per_audio = workspace / KEY_PER_AUDIO
    global_dir = workspace / KEY_GLOBAL

    return WorkspacePaths(
        workspace=workspace,
        per_audio=per_audio,
        global_dir=global_dir,
        clips=global_dir / KEY_CLIPS,
        evaluation=global_dir / KEY_EVALUATION,
    )


def ensure_workspace_dirs(datasets: List[DatasetPaths]) -> None:
    """
    Create workspace directory structure.
    """
    for ds in datasets:
        ws = get_workspace_paths(ds.workspace)

        ws.workspace.mkdir(parents=True, exist_ok=True)
        ws.per_audio.mkdir(parents=True, exist_ok=True)
        ws.global_dir.mkdir(parents=True, exist_ok=True)
        ws.clips.mkdir(parents=True, exist_ok=True)
        ws.evaluation.mkdir(parents=True, exist_ok=True)

# --- Pipeline Model Artifact Paths ---
def default_uvr_model_path(config: object) -> Path:
    """
    Derive the default UVR model path (kept internal for now).
    """
    # expects config.project.project_root
    return config.project.project_root / "models" / "UVR-MDX-NET-Inst_3.onnx"

# --- Workspace subtree helpers ---

def get_audio_root(workspace: Path, audio_id: str) -> Path:
    return get_workspace_paths(workspace).per_audio / audio_id


def get_audio_files_dir(workspace: Path, audio_id: str) -> Path:
    return get_audio_root(workspace, audio_id) / KEY_AUDIO_FILES


def get_annotations_dir(workspace: Path, audio_id: str) -> Path:
    return get_audio_root(workspace, audio_id) / KEY_ANNOTATIONS


def get_vad_dir(workspace: Path, audio_id: str) -> Path:
    return get_annotations_dir(workspace, audio_id) / KEY_VAD


def get_asr_dir(workspace: Path, audio_id: str) -> Path:
    return get_annotations_dir(workspace, audio_id) / KEY_ASR


def get_nlp_dir(workspace: Path, audio_id: str) -> Path:
    return get_annotations_dir(workspace, audio_id) / KEY_NLP


def get_nvv_dir(workspace: Path, audio_id: str) -> Path:
    return get_annotations_dir(workspace, audio_id) / KEY_NVV


def get_labels_dir(workspace: Path, audio_id: str) -> Path:
    return get_audio_root(workspace, audio_id) / KEY_LABELS


def get_vad_labels_dir(workspace: Path, audio_id: str) -> Path:
    return get_labels_dir(workspace, audio_id) / KEY_VAD


def get_asr_labels_dir(workspace: Path, audio_id: str) -> Path:
    return get_labels_dir(workspace, audio_id) / KEY_ASR


def get_nvv_labels_dir(workspace: Path, audio_id: str) -> Path:
    return get_labels_dir(workspace, audio_id) / KEY_NVV


def get_global_dir(workspace: Path) -> Path:
    return get_workspace_paths(workspace).global_dir


def get_global_clips_dir(workspace: Path) -> Path:
    return get_workspace_paths(workspace).clips


def get_global_clips_mode_dir(workspace: Path, gt_mode: str) -> Path:
    return get_global_clips_dir(workspace) / gt_mode


def get_global_evaluation_dir(workspace: Path) -> Path:
    return get_workspace_paths(workspace).evaluation


def get_global_evaluation_mode_dir(workspace: Path, gt_mode: str) -> Path:
    return get_global_evaluation_dir(workspace) / gt_mode


def get_research_questions_dir(workspace: Path, gt_mode: str) -> Path:
    return get_global_evaluation_mode_dir(workspace, gt_mode) / KEY_RQ


# --- Canonical artifact paths ---

def get_metadata_path(workspace: Path, audio_id: str) -> Path:
    return get_audio_root(workspace, audio_id) / f"{audio_id}_{KEY_METADATA}{EXT_JSON}"


def get_std_audio_path(workspace: Path, audio_id: str) -> Path:
    return get_audio_files_dir(workspace, audio_id) / f"{audio_id}_{KEY_STD}{EXT_WAV}"


def get_std_vocals_audio_path(workspace: Path, audio_id: str) -> Path:
    return get_audio_files_dir(workspace, audio_id) / f"{audio_id}_{KEY_STD}_{KEY_VOCALS}{EXT_WAV}"


def get_std_background_audio_path(workspace: Path, audio_id: str) -> Path:
    return get_audio_files_dir(workspace, audio_id) / f"{audio_id}_{KEY_STD}_{KEY_BACKGROUND}{EXT_WAV}"


def get_std_vocals_norm_audio_path(workspace: Path, audio_id: str) -> Path:
    return get_audio_files_dir(workspace, audio_id) / f"{audio_id}_{KEY_STD}_{KEY_VOCALS}_{KEY_NORM}{EXT_WAV}"


def get_std_background_norm_audio_path(workspace: Path, audio_id: str) -> Path:
    return get_audio_files_dir(workspace, audio_id) / f"{audio_id}_{KEY_STD}_{KEY_BACKGROUND}_{KEY_NORM}{EXT_WAV}"


def get_vad_json_path(workspace: Path, audio_id: str, audio_input: str) -> Path:
    return get_vad_dir(workspace, audio_id) / f"{audio_id}_{audio_input}_{KEY_VAD}{EXT_JSON}"


def get_vad_label_path(workspace: Path, audio_id: str, audio_input: str) -> Path:
    return get_vad_labels_dir(workspace, audio_id) / f"{audio_id}_{audio_input}_{KEY_VAD}{EXT_TXT}"


def get_asr_json_path(workspace: Path, audio_id: str, vad_mask: str, asr_audio_in: str) -> Path:
    return get_asr_dir(workspace, audio_id) / f"{audio_id}_{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}{EXT_JSON}"


def get_asr_label_path(workspace: Path, audio_id: str, vad_mask: str, asr_audio_in: str) -> Path:
    return get_asr_labels_dir(workspace, audio_id) / f"{audio_id}_{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}{EXT_TXT}"


def get_nlp_json_path(workspace: Path, audio_id: str, vad_mask: str, asr_audio_in: str) -> Path:
    return get_nlp_dir(workspace, audio_id) / f"{audio_id}_{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}_{KEY_NLP}{EXT_JSON}"


#legacy. toDo: eliminate combo_key usage, use get_nlp_json_path instead
def get_nvv_json_path_from_combo_key(workspace: Path, audio_id: str, combo_key: str) -> Path:
    return get_nvv_dir(workspace, audio_id) / f"{audio_id}_{combo_key}{EXT_JSON}"

def get_nvv_json_path(workspace: Path, audio_id: str, vad_mask: str, asr_audio_in: str) -> Path:
    return get_nvv_dir(workspace, audio_id) / f"{audio_id}_{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}_{KEY_NLP}_{KEY_NVV}{EXT_JSON}"


def get_nvv_label_path(workspace: Path, audio_id: str, vad_mask: str, asr_audio_in: str) -> Path:
    return get_nvv_labels_dir(workspace, audio_id) / f"{audio_id}_{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}_{KEY_NLP}_{KEY_NVV}{EXT_TXT}"


def get_nlp_log_json_path(workspace: Path, audio_id: str, vad_mask: str, asr_audio_in: str) -> Path:
    """Return canonical NLP log JSON path for a given combo."""
    return get_nlp_dir(workspace, audio_id) / f"{audio_id}_{vad_mask}_{KEY_VAD}_{asr_audio_in}_{KEY_ASR}_{KEY_NLP}_log{EXT_JSON}"


# --- Run tracking paths ---

def get_run_json_path(workspace: Path) -> Path:
    """Return canonical run.json path inside a workspace."""
    return Path(workspace) / _RUN_JSON


def get_runs_index_json_path(processed_root: Path) -> Path:
    """Return canonical runs_index.json path at the processed root level."""
    return Path(processed_root) / _RUNS_INDEX_JSON


# --- Per-audio evaluation paths ---

def get_per_audio_evaluation_dir(workspace: Path, audio_id: str) -> Path:
    """Return per-audio evaluation root: per_audio/<audio_id>/evaluation/."""
    return get_audio_root(workspace, audio_id) / KEY_EVALUATION


def get_per_audio_evaluation_mode_dir(workspace: Path, audio_id: str, mode: str) -> Path:
    """Return per-audio evaluation mode dir: per_audio/<audio_id>/evaluation/<mode>/."""
    return get_per_audio_evaluation_dir(workspace, audio_id) / mode


def get_per_audio_detailed_csv_path(workspace: Path, audio_id: str, mode: str) -> Path:
    """Return canonical per-audio detailed evaluation CSV path."""
    return get_per_audio_evaluation_mode_dir(workspace, audio_id, mode) / f"{audio_id}_detailed_{mode}.csv"


def get_per_audio_summary_csv_path(workspace: Path, audio_id: str, mode: str) -> Path:
    """Return canonical per-audio summary evaluation CSV path."""
    return get_per_audio_evaluation_mode_dir(workspace, audio_id, mode) / f"{audio_id}_summary_{mode}.csv"


def get_per_audio_evaluation_xlsx_path(workspace: Path, audio_id: str, mode: str) -> Path:
    """Return canonical per-audio evaluation XLSX path."""
    return get_per_audio_evaluation_mode_dir(workspace, audio_id, mode) / f"{audio_id}_evaluation_{mode}.xlsx"


# --- Global evaluation mode-level artifact paths ---
# evaluation_dir = workspace/global/evaluation

def get_eval_mode_dir(evaluation_dir: Path, mode: str) -> Path:
    """Return the mode-scoped directory inside an evaluation dir."""
    return Path(evaluation_dir) / mode


def get_detailed_all_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical detailed_all CSV path for the given evaluation mode."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"detailed_all_{mode}.csv"


def get_summary_all_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical summary_all CSV path for the given evaluation mode."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"summary_all_{mode}.csv"


def get_global_evaluation_xlsx_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global evaluation XLSX path for the given mode."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"{_GLOBAL_EVALUATION_XLSX_BASE}_{mode}.xlsx"


def get_global_combo_ranking_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global combo ranking CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_combo_ranking_{mode}.csv"


def get_global_best_k_set_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global best-k set CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_best_k_set_{mode}.csv"


def get_global_best_k_trace_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global best-k trace CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_best_k_trace_{mode}.csv"


def get_global_f1_vs_k_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global F1-vs-k CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_f1_vs_k_{mode}.csv"


def get_global_best_k_union_per_audio_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global best-k union per-audio CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_best_k_union_per_audio_{mode}.csv"


def get_global_best_k_union_summary_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global best-k union summary CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_best_k_union_summary_{mode}.csv"


def get_global_best_k_union_set_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical global best-k union set CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"global_best_k_union_set_{mode}.csv"


def get_pipeline_capability_summary_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical pipeline capability summary CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"pipeline_capability_{mode}_summary.csv"


def get_pipeline_capability_per_audio_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical pipeline capability per-audio CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"pipeline_capability_{mode}_per_audio.csv"


def get_pipeline_capability_xlsx_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical pipeline capability XLSX path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"{_PIPELINE_CAPABILITY_XLSX_BASE}_{mode}.xlsx"


def get_pipeline_capability_nvv_events_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical pipeline capability NVV events CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"pipeline_capability_{mode}_nvv_events.csv"


def get_rq2_audio_derivatives_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ2b audio derivatives CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"rq2_config_audio_derivatives_{mode}.csv"


def get_rq3_nvv_coverage_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ3 NVV coverage CSV path."""
    return get_eval_mode_dir(evaluation_dir, mode) / f"rq3_nvv_coverage_{mode}.csv"


# --- Research question output paths ---

def get_rq_output_dir(evaluation_dir: Path, mode: str) -> Path:
    """Return research questions output dir: <evaluation_dir>/<mode>/research_questions/."""
    return get_eval_mode_dir(evaluation_dir, mode) / KEY_RQ


def get_rq1_pipeline_capability_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ1 pipeline capability CSV path in research_questions/."""
    return get_rq_output_dir(evaluation_dir, mode) / f"rq1_pipeline_capability_{mode}.csv"


def get_rq2_config_ranking_single_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ2a single-config ranking CSV path in research_questions/."""
    return get_rq_output_dir(evaluation_dir, mode) / f"rq2_config_ranking_single_{mode}.csv"


def get_rq2_config_ranking_selected_set_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ2a selected-set ranking CSV path in research_questions/."""
    return get_rq_output_dir(evaluation_dir, mode) / f"rq2_config_ranking_selected_set_{mode}.csv"


def get_rq2_config_audio_derivatives_rq_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ2b audio derivatives CSV path in research_questions/."""
    return get_rq_output_dir(evaluation_dir, mode) / f"rq2_config_audio_derivatives_{mode}.csv"


def get_rq3_nvv_coverage_rq_csv_path(evaluation_dir: Path, mode: str) -> Path:
    """Return canonical RQ3 NVV coverage CSV path in research_questions/."""
    return get_rq_output_dir(evaluation_dir, mode) / f"rq3_nvv_coverage_{mode}.csv"


# --- Experiment evaluation paths ---

def get_experiment_evaluation_dir(experiment_root: Path) -> Path:
    """Return the evaluation directory inside an experiment root."""
    return Path(experiment_root) / KEY_EVALUATION


def get_experiment_eval_mode_dir(experiment_root: Path, mode: str) -> Path:
    """Return the mode-scoped evaluation directory inside an experiment root."""
    return get_experiment_evaluation_dir(experiment_root) / mode


def get_experiment_rq_output_dir(experiment_root: Path, mode: str) -> Path:
    """
    Return the experiment-level research questions output directory.

    Path: <experiment_root>/evaluation/<mode>/research_questions/
    """
    return get_experiment_eval_mode_dir(experiment_root, mode) / KEY_RQ


def get_experiment_rq1_csv_path(experiment_root: Path, mode: str) -> Path:
    """Return experiment-level RQ1 pipeline capability CSV path."""
    return get_experiment_rq_output_dir(experiment_root, mode) / f"rq1_pipeline_capability_experiment_{mode}.csv"


def get_experiment_rq2a_single_csv_path(experiment_root: Path, mode: str) -> Path:
    """Return experiment-level RQ2a single-config ranking CSV path."""
    return get_experiment_rq_output_dir(experiment_root, mode) / f"rq2_config_ranking_single_experiment_{mode}.csv"


def get_experiment_rq2a_selected_set_csv_path(experiment_root: Path, mode: str) -> Path:
    """Return experiment-level RQ2a selected-set ranking CSV path."""
    return get_experiment_rq_output_dir(experiment_root, mode) / f"rq2_config_ranking_selected_set_experiment_{mode}.csv"


def get_experiment_rq2b_csv_path(experiment_root: Path, mode: str) -> Path:
    """Return experiment-level RQ2b audio derivatives CSV path."""
    return get_experiment_rq_output_dir(experiment_root, mode) / f"rq2_config_audio_derivatives_experiment_{mode}.csv"


def get_experiment_rq3_csv_path(experiment_root: Path, mode: str) -> Path:
    """Return experiment-level RQ3 NVV coverage CSV path."""
    return get_experiment_rq_output_dir(experiment_root, mode) / f"rq3_nvv_coverage_experiment_{mode}.csv"


# --- Clip export subdir naming ---

def get_exploration_clips_sub_dir(subfolder: str, gt_mode: str, vad_mask: str, asr_audio_in: str) -> str:
    """
    Return the exploration clip export subdirectory name.

    Convention: exploration_<clip_mode>/<vad_mask>_vad__<asr_audio_in>_asr
    The double underscore between vad and asr_audio_in is intentional.
    """
    return f"{subfolder}_{gt_mode}/{vad_mask}_{KEY_VAD}__{asr_audio_in}_{KEY_ASR}"


def get_selected_set_clips_sub_dir(gt_mode: str) -> str:
    """Return the selected-set clip export subdirectory name."""
    return f"selected_set_nvvs_{gt_mode}"


# --- GT Units (YAML-driven) ---

def get_gt_units(
    cfg_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> List[GtExcelUnitPaths]:
    """
    Resolve GT units strictly based on evaluation.gt_units.
    No recursive search, no auto-derivation.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)
    project = project or get_project_paths(cfg_path)

    e = cfg.get("evaluation", {})

    gt_units_cfg = e.get("gt_units", [])
    if not gt_units_cfg:
        raise ValueError("evaluation.gt_units is empty or missing in YAML.")

    id_column = str(e["gt_id_column"])
    gt_mode = str(e["gt_mode"])
    labels_export_rel = str(e["gt_labels_export_rel"])

    units: List[GtExcelUnitPaths] = []

    for unit_cfg in gt_units_cfg:
        name = str(unit_cfg["name"])

        # RAW excel
        raw_excel_rel_path = str(unit_cfg["raw_excel_rel_path"])
        raw_excel_path = (project.raw_root / raw_excel_rel_path).resolve()

        # CLEANED excel (processed)
        cleaned_excel_rel_path = str(unit_cfg["cleaned_excel_rel_path"])
        cleaned_excel_path = (project.processed_root / cleaned_excel_rel_path).resolve()

        # LABEL export dir (processed/<gt_labels_export_rel>/<unit.name>)
        labels_export_dir = (
            project.processed_root / labels_export_rel / name
        ).resolve()

        # Optional VOCALS dataset root (RAW)
        vocals_dataset_root = None
        if "vocals_dataset_root_rel" in unit_cfg:
            vocals_dataset_root = (
                project.raw_root / str(unit_cfg["vocals_dataset_root_rel"])
            ).resolve()

        # Optional subset copy (RAW)
        vocals_subset_copy_dir = None
        if "vocals_subset_copy_rel" in unit_cfg:
            vocals_subset_copy_dir = (
                project.raw_root / str(unit_cfg["vocals_subset_copy_rel"])
            ).resolve()

        units.append(
            GtExcelUnitPaths(
                name=name,
                raw_excel_path=raw_excel_path,
                vocals_dataset_root=vocals_dataset_root,
                cleaned_excel_path=cleaned_excel_path,
                labels_export_dir=labels_export_dir,
                vocals_subset_copy_dir=vocals_subset_copy_dir,
                id_column=id_column,
                gt_mode=gt_mode,
            )
        )

    return units



def get_gt_truth_excel_paths(
    cfg_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> List[Path]:
    """
    Central Truth resolution for evaluation.

    If gt_merged_cleaned_excel_rel is set -> use only that.
    Otherwise -> use cleaned_excel_path of each unit.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)
    project = project or get_project_paths(cfg_path)

    e = cfg.get("evaluation", {})

    merged_rel = e.get("gt_merged_cleaned_excel_rel")

    if merged_rel:
        merged_path = (project.processed_root / str(merged_rel)).resolve()
        return [merged_path]

    units = get_gt_units(cfg_path, project=project)
    return [u.cleaned_excel_path for u in units]


def get_gt_merged_cleaned_excel_path(
    cfg_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> Optional[Path]:
    """
    Return the configured merged cleaned GT excel output path (processed),
    or None if no merged path is configured.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)
    project = project or get_project_paths(cfg_path)

    merged_rel = cfg.get("evaluation", {}).get("gt_merged_cleaned_excel_rel")
    if merged_rel is None:
        return None

    merged_rel = str(merged_rel).strip()
    if merged_rel == "":
        return None

    return (project.processed_root / merged_rel).resolve()


def ensure_gt_dirs(
    cfg_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> None:
    """
    Create all directories implied by evaluation YAML:
    - cleaned excel parent dirs
    - labels export dirs
    - optional subset copy dirs
    - optional merged cleaned parent dir
    """
    cfg_path = Path(cfg_path).resolve()
    project = project or get_project_paths(cfg_path)
    cfg = _load_yaml(cfg_path)

    units = get_gt_units(cfg_path, project=project)

    for u in units:
        u.cleaned_excel_path.parent.mkdir(parents=True, exist_ok=True)
        u.labels_export_dir.mkdir(parents=True, exist_ok=True)

        if u.vocals_subset_copy_dir is not None:
            u.vocals_subset_copy_dir.mkdir(parents=True, exist_ok=True)

    merged_rel = cfg.get("evaluation", {}).get("gt_merged_cleaned_excel_rel")
    if merged_rel:
        merged_path = (project.processed_root / str(merged_rel)).resolve()
        merged_path.parent.mkdir(parents=True, exist_ok=True)


# --- Evaluation Paths ---

def get_evaluation_paths(
    cfg_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> EvaluationPaths:
    """
    Resolve:
    - GT units
    - dataset result folders
    - truth excel list (central resolution)
    """
    cfg_path = Path(cfg_path).resolve()
    project = project or get_project_paths(cfg_path)

    gt_units = get_gt_units(cfg_path, project=project)
    truth_excels = get_gt_truth_excel_paths(cfg_path, project=project)

    datasets = get_datasets(cfg_path, project=project)

    dataset_results: List[EvaluationDatasetPaths] = []

    for ds in datasets:
        ws = get_workspace_paths(ds.workspace)

        dataset_results.append(
            EvaluationDatasetPaths(
                name=ds.name,
                workspace=ws.workspace,
                results_dir=ws.evaluation,
            )
        )

    return EvaluationPaths(
        gt_units=gt_units,
        dataset_results=dataset_results,
        truth_excels=truth_excels,
    )


def get_experiment_run_root(
    cfg_path: str | Path,
    experiment_yaml_path: str | Path,
    project: Optional[ProjectPaths] = None,
) -> Path:
    """
    Resolve the executed experiment run root.

    Current convention:
        <processed_root>/<experiment_name>

    Args:
        cfg_path: Base config YAML path.
        experiment_yaml_path: Experiment definition YAML path.
        project: Optional pre-resolved project paths.

    Returns:
        Path to the experiment run root that contains resolved
        '*_config.yaml' files.

    Raises:
        FileNotFoundError: If the resolved experiment run root does not exist.
        ValueError: If the experiment name is missing or invalid.
    """
    cfg_path = Path(cfg_path).resolve()
    experiment_yaml_path = Path(experiment_yaml_path).resolve()

    project = project or get_project_paths(cfg_path)
    exp = _load_yaml(experiment_yaml_path)

    experiment_name = str(exp.get("experiment", experiment_yaml_path.stem)).strip()
    if not experiment_name:
        raise ValueError("Experiment name is empty.")

    experiment_run_root = (project.processed_root / experiment_name).resolve()

    if not experiment_run_root.exists():
        raise FileNotFoundError(
            f"Resolved experiment run root not found: {experiment_run_root}"
        )

    return experiment_run_root


# --- Debug Print ---

def print_paths(cfg_path: str | Path) -> None:
    """
    Print all resolved project, GT and evaluation paths.
    """
    cfg_path = Path(cfg_path).resolve()

    project = get_project_paths(cfg_path)
    datasets = get_datasets(cfg_path, project=project)
    units = get_gt_units(cfg_path, project=project)
    truth = get_gt_truth_excel_paths(cfg_path, project=project)

    print("\n" + "=" * 80)
    print("NVV PATH CONFIG PRINT")
    print("=" * 80)

    print("\n[PROJECT]")
    print(f"  cfg_path:       {cfg_path}")
    print(f"  project_root:   {project.project_root}")
    print(f"  raw_root:       {project.raw_root}")
    print(f"  processed_root: {project.processed_root}")

    print("\n[PIPELINE WORKSPACES]")
    for ds in datasets:
        ws = get_workspace_paths(ds.workspace)
        print(f"\n  - {ds.name}")
        print(f"      input_dir:  {ds.input_dir}")
        print(f"      workspace:  {ws.workspace}")
        print(f"      evaluation: {ws.evaluation}")

    print("\n[GROUND TRUTH UNITS]")
    for u in units:
        print(f"\n  - {u.name}")
        print(f"      raw_excel:          {u.raw_excel_path}")
        print(f"      cleaned_excel:      {u.cleaned_excel_path}")
        print(f"      labels_export_dir:  {u.labels_export_dir}")
        print(f"      vocals_dataset_root:{u.vocals_dataset_root}")
        print(f"      vocals_subset_copy: {u.vocals_subset_copy_dir}")
        print(f"      id_column:          {u.id_column}")
        print(f"      gt_mode:            {u.gt_mode}")

    print("\n[TRUTH GT FILES]")
    for p in truth:
        print(f"  - {p}")

    print("\n" + "=" * 80)
