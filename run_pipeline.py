#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the reduced pipeline with Step 1 and Step 2 only.

CLI usage:
    python run_pipeline.py --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config.load_config import ensure_workspace, print_config
from config.path_factory import default_uvr_model_path
from pipeline.step_1_standardize import run_step_1_std
from pipeline.step_2_separate import run_step_2_separate


def run_pipeline_from_config(
    config: object,
    *,
    model_path: Optional[Path] = None,
) -> None:
    """
    Run Step 1 and Step 2 for all configured datasets.
    """
    ensure_workspace(config)
    print_config(config)

    if model_path is None:
        model_path = default_uvr_model_path(config)

    print(f"Using UVR model from: {model_path}")

    for ds in config.datasets:
        print(f"\n🚀 Step 1 Standardization for {ds.name}, Input: {ds.input_dir}, Output: {ds.workspace}.")
        run_step_1_std(
            input_dir=ds.input_dir,
            workspace=ds.workspace,
            device=config.runtime.device,
            force=config.runtime.force,
        )
    print("\n✅ Step 1 Standardization completed for all datasets.")

    for ds in config.datasets:
        print(f"\n🚀 Step 2 Source Separation for {ds.name} in {ds.workspace}.")
        run_step_2_separate(
            workspace=ds.workspace,
            model_path=model_path,
            device=config.runtime.device,
            force=config.runtime.force,
        )
    print("\n✅ Step 2 Source Separation completed for all datasets.")


def main() -> None:
    p = argparse.ArgumentParser(description="Run Step 1 and Step 2 of the audio pipeline")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to config.yaml (e.g., ./config/config.yaml)",
    )
    args = p.parse_args()

    config_path = Path(args.config).resolve()

    from config.load_config import load_config

    config = load_config(config_path)
    run_pipeline_from_config(config)


if __name__ == "__main__":
    main()
