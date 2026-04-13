from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.path_factory import ProjectPaths, DatasetPaths, ensure_workspace_dirs, get_datasets, get_project_paths


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    force: bool


@dataclass(frozen=True)
class AppConfig:
    """
    Reduced config for Step 1 and Step 2 only.
    """

    cfg_path: Path
    project: ProjectPaths
    runtime: RuntimeConfig
    datasets: list[DatasetPaths]

    def print_datasets(self) -> None:
        for ds in self.datasets:
            print(f"\n▶ Dataset: {ds.name}\n  Input: {ds.input_dir}\n  Workspace: {ds.workspace}")


def _load_yaml(cfg_path: Path) -> dict[str, Any]:
    import yaml

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(cfg_path: str | Path) -> AppConfig:
    """
    Load config for Step 1 and Step 2 only.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)

    project = get_project_paths(cfg_path)
    datasets = get_datasets(cfg_path, project=project)

    runtime_cfg = cfg.get("runtime", {}) or {}
    runtime = RuntimeConfig(
        device=str(runtime_cfg.get("device", "auto")),
        force=bool(runtime_cfg.get("force", False)),
    )

    return AppConfig(
        cfg_path=cfg_path,
        project=project,
        runtime=runtime,
        datasets=datasets,
    )


def ensure_workspace(cfg: AppConfig) -> None:
    ensure_workspace_dirs(cfg.datasets)


def print_config(cfg: AppConfig) -> None:
    print("\n" + "=" * 80)
    print("STEP 1 / STEP 2 CONFIG")
    print("=" * 80)

    print("\n[PROJECT]")
    print(f"  cfg_path:       {cfg.cfg_path}")
    print(f"  project_root:   {cfg.project.project_root}")
    print(f"  raw_root:       {cfg.project.raw_root}")
    print(f"  processed_root: {cfg.project.processed_root}")

    print("\n[RUNTIME]")
    print(f"  device: {cfg.runtime.device}")
    print(f"  force:  {cfg.runtime.force}")

    print("\n[DATASETS]")
    for i, ds in enumerate(cfg.datasets, start=1):
        print(f"  ({i}) name:      {ds.name}")
        print(f"      input_dir:  {ds.input_dir}")
        print(f"      workspace:  {ds.workspace}")

    print("\n" + "=" * 80 + "\n")
