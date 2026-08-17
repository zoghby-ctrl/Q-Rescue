"""Bit-faithful replay of accepted Step 9 synthetic observation features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def regenerate_step9_observations(
    effective_config_path: str | Path,
    saved_observations_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Regenerate pre-CSV floats and verify them against the accepted table.

    Decimal CSV round trips can move values across fitted histogram-tree split
    thresholds even when their absolute change is near machine precision.
    Callers must separately hash-verify the accepted Step 9 runner and Python
    dependencies before trusting this deterministic replay.
    """

    from experiments.run_step9_cross_network import (  # noqa: PLC0415
        _build_networks,
        _generate_observation_partitions,
    )

    config_path = Path(effective_config_path)
    saved_path = Path(saved_observations_path)
    step9_config = json.loads(config_path.read_text(encoding="utf-8"))
    networks, registry = _build_networks(step9_config)
    frames, _, _ = _generate_observation_partitions(
        step9_config, networks, registry
    )
    ordered_keys = (
        "train",
        "validation",
        "calibration",
        "test_id",
        "ood_topology",
        "ood_traffic",
        "ood_combined",
    )
    regenerated = pd.concat(
        [frames[key] for key in ordered_keys], ignore_index=True
    )
    saved = pd.read_csv(saved_path)
    if len(saved) != len(regenerated):
        raise RuntimeError("regenerated Step 9 observation count differs from CSV")
    if saved["observation_id"].duplicated().any() or regenerated[
        "observation_id"
    ].duplicated().any():
        raise RuntimeError("Step 9 observation IDs are not unique")
    regenerated = (
        regenerated.set_index("observation_id")
        .loc[saved["observation_id"]]
        .reset_index()
    )
    metadata_columns = [
        "observation_id",
        "dataset_group",
        "network_id",
        "topology",
        "scenario",
    ]
    metadata_match = all(
        saved[column].astype(str).equals(regenerated[column].astype(str))
        for column in metadata_columns
    )
    if not metadata_match:
        raise RuntimeError("regenerated Step 9 observation metadata differs from CSV")
    numeric_columns = sorted(
        set(saved.select_dtypes(include=[np.number]).columns)
        & set(regenerated.select_dtypes(include=[np.number]).columns)
    )
    numeric_delta = np.abs(
        saved[numeric_columns].to_numpy(dtype=float)
        - regenerated[numeric_columns].to_numpy(dtype=float)
    )
    maximum_delta = float(np.max(numeric_delta))
    if maximum_delta > 1e-12:
        raise RuntimeError(
            f"regenerated Step 9 numeric data differ materially: {maximum_delta}"
        )
    checks: dict[str, object] = {
        "observation_count": int(len(regenerated)),
        "observation_id_and_metadata_exact_match": metadata_match,
        "numeric_column_count_compared": int(len(numeric_columns)),
        "max_absolute_numeric_csv_roundtrip_delta": maximum_delta,
        "nonzero_numeric_csv_roundtrip_delta_count": int(
            np.count_nonzero(numeric_delta)
        ),
        "reconstruction_reason": (
            "decimal CSV round trips can cross fitted histogram-tree split "
            "thresholds; accepted Step 9 Python sources and config must be hash-verified"
        ),
    }
    return regenerated, checks
