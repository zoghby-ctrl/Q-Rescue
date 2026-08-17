"""Run final tests/link/hash audits and write the immutable freeze inventory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SCIENCE_FREEZE_MANIFEST.json"
EXPECTED_PYTEST_PASS_COUNT = 158
FINAL_RESULT_DIRECTORIES = [
    "step9_2_topology_uncertainty_qaoa",
    "step9_6_statistics",
    "step9_6b_topology_mechanisms",
    "step9_7_adaptive_conformal",
    "step9_8_noise",
    "targeted_k16_scaling",
    "standard_x_penalty_sensitivity",
]
FREEZE_DOCUMENTS = [
    "SCIENTIFIC_AUDIT.md",
    "STEP_9_2_FINDINGS.md",
    "FINAL_SCIENCE_FREEZE.md",
    "FINAL_RESEARCH_STATUS.md",
    "TA_PROGRESS.md",
    "EXPERIMENT_INDEX.md",
    "FIGURE_INDEX.md",
    "TABLE_INDEX.md",
    "paper/ABSTRACT_DRAFT.md",
    "paper/INTRODUCTION.md",
    "paper/RELATED_WORK.md",
    "paper/evidence/LITERATURE_MASTER_MATRIX.md",
    "paper/evidence/LITERATURE_QUANTUM_MATRIX.md",
    "paper/evidence/LITERATURE_UNCERTAINTY_GENERALIZATION_MATRIX.md",
    "paper/PROBLEM_FORMULATION.md",
    "paper/METHOD.md",
    "paper/EXPERIMENTAL_SETUP.md",
    "paper/RESULTS.md",
    "paper/CLAIMS.md",
    "paper/LIMITATIONS.md",
    "paper/REPRODUCIBILITY.md",
    "paper/MANUSCRIPT_DRAFT.md",
    "paper/MANUSCRIPT_DRAFT_MANIFEST.json",
    "paper/figures/science_freeze_figure_manifest.json",
    "paper/figures/step9_5_figure_manifest.json",
    "paper/tables/science_freeze_table_manifest.json",
]
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_mapping(directory: Path, mapping: dict[str, str]) -> int:
    checked = 0
    for name, expected in mapping.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"manifest target missing: {path}")
        if sha256(path) != expected:
            raise RuntimeError(f"manifest hash mismatch: {path}")
        checked += 1
    return checked


def verify_result_manifests() -> dict[str, object]:
    hashes: dict[str, str] = {}
    artifact_count = 0
    checkpoint_count = 0
    for name in FINAL_RESULT_DIRECTORIES:
        directory = ROOT / "results" / "research" / name
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"missing final result manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_count += verify_mapping(
            directory, dict(manifest.get("artifact_sha256", {}))
        )
        checkpoint_hashes = dict(manifest.get("checkpoint_sha256", {}))
        if checkpoint_hashes:
            checkpoint_count += verify_mapping(
                directory / "checkpoints", checkpoint_hashes
            )
        prior_failures = dict(manifest.get("prior_failed_attempt_sha256", {}))
        if prior_failures:
            checkpoint_count += verify_mapping(
                directory / "failures", prior_failures
            )
        hashes[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)

    step92 = ROOT / "results/research/step9_2_topology_uncertainty_qaoa"
    checkpoint_manifest = json.loads(
        (step92 / "checkpoint_hashes.json").read_text(encoding="utf-8")
    )
    trial_hashes = dict(checkpoint_manifest["trial_sha256"])
    if checkpoint_manifest["trial_count"] != 1800 or len(trial_hashes) != 1800:
        raise RuntimeError("Step 9.2 checkpoint hash inventory is not 1,800 trials")
    checkpoint_count += verify_mapping(step92 / "checkpoints" / "trials", trial_hashes)
    return {
        "manifest_sha256": hashes,
        "verified_artifact_count": artifact_count,
        "verified_checkpoint_or_failure_count": checkpoint_count,
    }


def verify_paper_manifests() -> dict[str, object]:
    legacy_figure_manifest_path = ROOT / "paper/figures/step9_5_figure_manifest.json"
    figure_manifest_path = ROOT / "paper/figures/science_freeze_figure_manifest.json"
    table_manifest_path = ROOT / "paper/tables/science_freeze_table_manifest.json"
    manuscript_manifest_path = ROOT / "paper/MANUSCRIPT_DRAFT_MANIFEST.json"
    for path in [legacy_figure_manifest_path, figure_manifest_path, table_manifest_path, manuscript_manifest_path]:
        if not path.is_file():
            raise RuntimeError(f"missing publication manifest: {path}")

    legacy_figures = json.loads(
        legacy_figure_manifest_path.read_text(encoding="utf-8")
    )
    figures = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    tables = json.loads(table_manifest_path.read_text(encoding="utf-8"))
    manuscript = json.loads(manuscript_manifest_path.read_text(encoding="utf-8"))
    legacy_figure_count = verify_mapping(
        legacy_figure_manifest_path.parent,
        dict(legacy_figures["artifact_sha256"]),
    )
    figure_count = verify_mapping(
        figure_manifest_path.parent, dict(figures["artifact_sha256"])
    )
    table_count = verify_mapping(
        table_manifest_path.parent, dict(tables["artifact_sha256"])
    )
    manuscript_path = ROOT / "paper/MANUSCRIPT_DRAFT.md"
    if sha256(manuscript_path) != manuscript["output_sha256"]:
        raise RuntimeError("manuscript output hash mismatch")
    abstract_path = ROOT / "paper/ABSTRACT_DRAFT.md"
    if sha256(abstract_path) != manuscript["abstract_output_sha256"]:
        raise RuntimeError("abstract output hash mismatch")
    if sha256(ROOT / "paper/MANUSCRIPT_TEMPLATE.md") != manuscript["template_sha256"]:
        raise RuntimeError("manuscript template hash mismatch")
    if sha256(ROOT / "paper/ABSTRACT_TEMPLATE.md") != manuscript["abstract_template_sha256"]:
        raise RuntimeError("abstract template hash mismatch")
    for manifest, builder in [
        (legacy_figures, ROOT / "experiments/build_step9_5_figures.py"),
        (figures, ROOT / "experiments/build_science_freeze_figures.py"),
        (tables, ROOT / "experiments/build_science_freeze_tables.py"),
        (manuscript, ROOT / "experiments/build_manuscript_draft.py"),
    ]:
        if sha256(builder) != manifest["builder_sha256"]:
            raise RuntimeError(f"publication builder hash mismatch: {builder}")
        for name, expected in dict(manifest["source_sha256"]).items():
            source = ROOT / name
            if not source.is_file() or sha256(source) != expected:
                raise RuntimeError(f"publication source hash mismatch: {source}")
    return {
        "figure_artifact_count": legacy_figure_count + figure_count,
        "table_artifact_count": table_count,
        "manuscript_numeric_token_count": manuscript["numeric_tokens_populated"],
        "manifest_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [legacy_figure_manifest_path, figure_manifest_path, table_manifest_path, manuscript_manifest_path]
        },
    }


def audit_links() -> dict[str, object]:
    checked = 0
    missing: list[dict[str, str]] = []
    markdown_files = sorted(ROOT.rglob("*.md"))
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if target.startswith(("http://", "https://", "mailto:", "app://", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            checked += 1
            candidate = Path(target)
            if not candidate.is_absolute():
                candidate = document.parent / candidate
            # The verifier creates this target only after the link audit passes.
            if candidate.resolve() == OUTPUT.resolve():
                continue
            if not candidate.exists():
                missing.append(
                    {
                        "document": str(document.relative_to(ROOT)),
                        "target": target,
                    }
                )
    if missing:
        raise RuntimeError(f"missing local Markdown links: {missing[:10]}")
    return {
        "markdown_file_count": len(markdown_files),
        "local_link_count": checked,
        "missing_link_count": 0,
    }


def run_tests() -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"(\d+) passed", combined)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(f"final pytest failed:\n{combined[-4000:]}")
    passed = int(match.group(1))
    if passed != EXPECTED_PYTEST_PASS_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_PYTEST_PASS_COUNT} passing tests, found {passed}"
        )
    return {
        "command": f"{sys.executable} -m pytest -q",
        "passed": passed,
        "exit_code": completed.returncode,
        "wall_time_seconds": elapsed,
        "warning_count": len(re.findall(r"Warning", combined)),
        "output_tail": combined[-2000:],
    }


def main() -> None:
    result_audit = verify_result_manifests()
    paper_audit = verify_paper_manifests()
    links = audit_links()
    tests = run_tests()
    master = (ROOT / "paper/evidence/LITERATURE_MASTER_MATRIX.md").read_text(
        encoding="utf-8"
    )
    literature_rows = len(
        re.findall(r"^\| \[[^\n]+\|$", master, flags=re.MULTILINE)
    )
    if literature_rows != 33:
        raise RuntimeError(f"expected 33 literature rows, found {literature_rows}")
    missing_docs = [name for name in FREEZE_DOCUMENTS if not (ROOT / name).is_file()]
    if missing_docs:
        raise RuntimeError(f"missing freeze documents: {missing_docs}")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256(Path(__file__)),
        "repository_root": str(ROOT),
        "git_commit": None,
        "git_state_available": False,
        "result_audit": result_audit,
        "publication_audit": paper_audit,
        "markdown_link_audit": links,
        "pytest": tests,
        "literature_substantive_row_count": literature_rows,
        "freeze_document_sha256": {
            name: sha256(ROOT / name) for name in FREEZE_DOCUMENTS
        },
        "claim_exclusions": [
            "quantum_advantage",
            "quantum_speedup",
            "hardware_result",
            "classical_routing_superiority",
            "observed_real_traffic_validation",
            "emergency_deployment_readiness",
        ],
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Science freeze verified: {tests['passed']} tests, "
        f"{links['local_link_count']} local links, "
        f"{result_audit['verified_artifact_count']} result artifacts"
    )


if __name__ == "__main__":
    main()
