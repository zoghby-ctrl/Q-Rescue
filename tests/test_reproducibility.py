from pathlib import Path

from src.reproducibility import (
    build_experiment_manifest,
    environment_metadata,
    file_sha256,
    write_json,
)


def test_file_hash_and_json_writer_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "record.json"
    write_json(output, {"b": 2, "a": 1})
    first_hash = file_sha256(output)
    write_json(output, {"b": 2, "a": 1})
    assert file_sha256(output) == first_hash
    assert output.read_text(encoding="utf-8").startswith('{\n  "a": 1')


def test_manifest_records_missing_git_commit(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"seed": 7}\n', encoding="utf-8")
    manifest = build_experiment_manifest(
        experiment_name="unit-test",
        repository_root=tmp_path,
        random_seeds={"data": 7},
        parameters={"alpha": 0.1},
        config_path=config,
    )
    assert manifest["git_commit"] is None
    assert manifest["random_seeds"] == {"data": 7}
    assert manifest["config_sha256"] == file_sha256(config)


def test_environment_metadata_contains_required_versions() -> None:
    metadata = environment_metadata()
    assert metadata["python_version"]
    assert metadata["packages"]["numpy"]
    assert metadata["packages"]["pennylane"] == "0.45.1"
