"""One-time online cache stage for the Step 8.5 real-topology benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.network_loader import load_network_graphml
from src.osm_adapter import (
    BoundingBox,
    OSM_ATTRIBUTION,
    OSM_COPYRIGHT_URL,
    OSMExtractionSpec,
    build_overpass_query,
    fetch_overpass_json,
    overpass_json_to_canonical_graph,
    save_cached_graphml,
)
from src.reproducibility import build_experiment_manifest, file_sha256, write_json


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "real_topology_benchmark.json"


def _spec(entry: dict[str, object], config: dict[str, object]) -> OSMExtractionSpec:
    bbox = dict(entry["bounding_box"])
    return OSMExtractionSpec(
        network_id=str(entry["network_id"]),
        region_label=str(entry["region_label"]),
        selection_rationale=str(entry["selection_rationale"]),
        topology_class=str(entry["topology_class"]),
        bounding_box=BoundingBox(
            float(bbox["south"]),
            float(bbox["west"]),
            float(bbox["north"]),
            float(bbox["east"]),
        ),
        highway_classes=tuple(str(value) for value in config["highway_classes"]),
        candidate_k=int(config["candidate_k"]),
    )


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def run(*, resume: bool = False) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    study_label = str(config["study_label"])
    cache_directory = REPOSITORY_ROOT / str(config["cache_directory"])
    cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = REPOSITORY_ROOT / str(config["cache_manifest"])
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"cache manifest already exists; refuse one-time overwrite: {manifest_path}"
        )

    rows: list[dict[str, object]] = []
    for entry_value in config["regions"]:
        entry = dict(entry_value)
        spec = _spec(entry, config)
        raw_path = cache_directory / f"{spec.network_id}.overpass.json"
        graph_path = cache_directory / f"{spec.network_id}.graphml"
        if raw_path.exists() != graph_path.exists():
            raise FileExistsError(f"incomplete cache pair for {spec.network_id}")

        if raw_path.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite {raw_path}")
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            graph = load_network_graphml(graph_path)
            successful_endpoint = str(graph.graph["overpass_endpoint"])
            cache_action = "verified_existing_without_download"
        else:
            print(f"Fetching {spec.network_id}: {spec.region_label}", flush=True)
            payload = None
            errors: list[str] = []
            successful_endpoint = ""
            endpoints = config.get(
                "overpass_endpoints", [str(config["overpass_endpoint"])]
            )
            for endpoint_value in endpoints:
                endpoint = str(endpoint_value)
                for attempt in (1, 2):
                    try:
                        payload = fetch_overpass_json(
                            spec, endpoint=endpoint, timeout_seconds=120.0
                        )
                        successful_endpoint = endpoint
                        break
                    except Exception as error:  # network/service failover is intentional here
                        errors.append(
                            f"{endpoint} attempt {attempt}: {type(error).__name__}: {error}"
                        )
                        print(errors[-1], flush=True)
                        time.sleep(1.0)
                if payload is not None:
                    break
            if payload is None:
                raise RuntimeError(
                    f"all Overpass endpoints failed for {spec.network_id}: {errors}"
                )
            raw_path.write_bytes(_canonical_json_bytes(payload))
            graph = overpass_json_to_canonical_graph(payload, spec)
            graph.graph["overpass_endpoint"] = successful_endpoint
            graph.graph["raw_overpass_sha256"] = file_sha256(raw_path)
            save_cached_graphml(graph, graph_path)
            cache_action = "downloaded_once_and_cached"

        if int(graph.graph["od_candidate_route_count"]) != int(config["candidate_k"]):
            raise RuntimeError(f"{spec.network_id} is not K={config['candidate_k']} complete")
        query = build_overpass_query(spec)
        rows.append(
            {
                "study_label": study_label,
                "network_id": spec.network_id,
                "topology_class": spec.topology_class,
                "region_label": spec.region_label,
                "selection_rationale": spec.selection_rationale,
                "bounding_box": spec.bounding_box.overpass_text(),
                "overpass_endpoint": successful_endpoint,
                "overpass_query": query,
                "overpass_query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "source_timestamp_utc": str(graph.graph["source_timestamp_utc"]),
                "cached_at_utc": str(graph.graph["cached_at_utc"]),
                "attribution": OSM_ATTRIBUTION,
                "copyright_url": OSM_COPYRIGHT_URL,
                "raw_file": raw_path.name,
                "raw_sha256": file_sha256(raw_path),
                "graphml_file": graph_path.name,
                "graphml_sha256": file_sha256(graph_path),
                "node_count": graph.number_of_nodes(),
                "directed_edge_count": graph.number_of_edges(),
                "origin": graph.graph["origin"],
                "destination": graph.graph["destination"],
                "candidate_k": int(config["candidate_k"]),
                "parallel_edges_collapsed": int(
                    graph.graph.get("parallel_edges_collapsed", 0)
                ),
                "parallel_edge_policy": str(graph.graph["parallel_edge_policy"]),
                "parallel_edge_projection_caveat": str(
                    graph.graph["parallel_edge_projection_caveat"]
                ),
                "cache_action": cache_action,
            }
        )
        print(
            f"Cached {spec.network_id}: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, K={graph.graph['od_candidate_route_count']}",
            flush=True,
        )

    registry = pd.DataFrame(rows).sort_values("network_id").reset_index(drop=True)
    if len(registry) != 4 or registry["topology_class"].nunique() != 4:
        raise RuntimeError("Step 8.5 cache requires four distinct topology motivations")
    registry_path = cache_directory / "cache_registry.csv"
    registry.to_csv(registry_path, index=False)

    manifest = build_experiment_manifest(
        experiment_name="step8_5_real_topology_cache",
        repository_root=REPOSITORY_ROOT,
        random_seeds={},
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "study_label": study_label,
            "network_access": "online_cache_stage_only",
            "cache_is_immutable_without_explicit_resume": True,
            "attribution": OSM_ATTRIBUTION,
            "copyright_url": OSM_COPYRIGHT_URL,
            "network_count": len(registry),
            "all_networks_k_complete": bool(
                (registry["candidate_k"] == int(config["candidate_k"])).all()
            ),
            "files": {
                row.network_id: {
                    "raw_file": row.raw_file,
                    "raw_sha256": row.raw_sha256,
                    "graphml_file": row.graphml_file,
                    "graphml_sha256": row.graphml_sha256,
                    "source_timestamp_utc": row.source_timestamp_utc,
                    "bounding_box": row.bounding_box,
                    "overpass_query": row.overpass_query,
                    "attribution": row.attribution,
                }
                for row in registry.itertuples(index=False)
            },
            "source_sha256": {
                "configs/real_topology_benchmark.json": file_sha256(CONFIG_PATH),
                "experiments/cache_step8_5_real_topologies.py": file_sha256(Path(__file__)),
                "src/osm_adapter.py": file_sha256(REPOSITORY_ROOT / "src" / "osm_adapter.py"),
            },
            "artifact_sha256": {registry_path.name: file_sha256(registry_path)},
        }
    )
    write_json(manifest_path, manifest)
    print(f"Cache manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify completed cache pairs and fetch only missing regions; never overwrite pairs",
    )
    arguments = parser.parse_args()
    run(resume=arguments.resume)
