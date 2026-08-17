"""Small helpers for traceable, reproducible research artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACKED_PACKAGES = (
    "pennylane",
    "numpy",
    "pandas",
    "networkx",
    "scikit-learn",
    "scipy",
    "matplotlib",
)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_or_none(repository_root: str | Path) -> str | None:
    """Return the current Git commit, or ``None`` outside a Git checkout."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repository_root),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def environment_metadata() -> dict[str, Any]:
    """Return runtime and directly relevant package versions."""

    package_versions: dict[str, str | None] = {}
    for package_name in TRACKED_PACKAGES:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = None

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions,
    }


def build_experiment_manifest(
    experiment_name: str,
    repository_root: str | Path,
    random_seeds: dict[str, int],
    parameters: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance record for one experiment."""

    root = Path(repository_root).resolve()
    manifest: dict[str, Any] = {
        "experiment_name": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "git_commit": git_commit_or_none(root),
        "random_seeds": dict(random_seeds),
        "parameters": parameters,
        "environment": environment_metadata(),
    }
    if config_path is not None:
        resolved_config = Path(config_path).resolve()
        manifest["config_path"] = str(resolved_config)
        manifest["config_sha256"] = file_sha256(resolved_config)
    return manifest


def write_json(path: str | Path, payload: Any) -> None:
    """Write stable, human-readable JSON, creating parent folders as needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

