# Step 8.5 real-topology validation report

## Scope

**real road topology with controlled simulated traffic.** This is not observed, historical, or live traffic. Four compact
OpenStreetMap extracts were downloaded once, cached with ODbL attribution and
source hashes, and replayed under an active no-network guard. Place labels and
bounding boxes are configuration data; no routing or learning branch uses a city name.

## Validation summary

- Cached networks: 4 across 4 topology motivations.
- K=4 complete networks: 4/4.
- Frozen Step 4 distance normalization: 12.081912130491 km, read from the original training split rather than recomputed on this real batch.
- Frozen point-model test MAE/RMSE/R2: 1.343 / 1.486 / -0.187.
- Frozen local conformal nominal/empirical coverage: 0.900 / 0.544.
- Risk decisions changed from expected-time selection: 0/8 variant decisions.
- Source fastest-time projections excluded from full-multigraph distance claims: 0/4 networks.
- Compact simulator QAOA runs: 8; feasible-XY validity range 1.000000-1.000000.

## Scientific interpretation

These results test empirical transfer to real road structure under a controlled
simulator. They do not establish real-traffic effectiveness, universal
cross-network generalization, quantum advantage, operational emergency benefit,
or hardware performance. Local conformal intervals are frozen from Step 5;
coverage on these shifted networks is measured rather than assumed.

## Projection policy

The OSM adapter projects parallel roads to the minimum-base-time edge required
by the canonical DiGraph. Every projection is counted and labeled. A graph-wide
distance result is omitted whenever that source projection could have removed a
shorter parallel road. The cached raw Overpass response remains the audit source.

## Reproducibility

Run the online cache stage once with
`python experiments/cache_step8_5_real_topologies.py`. Thereafter run this file.
It verifies raw/GraphML hashes and activates a guard that raises on attempted
network access. Artifact hashes and frozen-model hashes are in `manifest.json`.
No quantum hardware was used.
