# Validation Report — Step 9.1 Leave-One-Network-Out

## Audit status

- Status: **PASS** for the mechanics of this bounded synthetic LONO experiment.
- Corpus: 20 networks, four from each of five topology families; every network is held out exactly once.
- Each fold contains 9 whole training networks, 5 whole validation networks, 5 whole calibration networks, and 1 whole held-out network.
- Validation and calibration each contain exactly one remaining network per family in every fold.
- All 20 folds have zero pairwise network overlap and zero pairwise observation overlap.
- Original Step 9 partition labels are retained only as metadata. All 5 original-label groups appear across train, validation, calibration, and held-out roles; no original label is reused as a fixed role.
- Every individual network appears in all four roles across the rotation and exactly once as held-out.
- Normalization, point fitting, point selection, residual-scale fitting, conformal calibration, and evaluation use train, train, validation, validation, calibration, and held-out roles respectively.
- Alpha=0.10 is fixed in configuration before folds. There is no fold-wise alpha selection.
- Fold-local distance divisors range from 7.4748 to 11.4788 km. No clipping is used; 24 held-out rows exceed normalized distance 1.0.

## Point prediction aggregate

- Pooled held-out rows: 4320.
- Pooled MAE: 0.8930 minutes.
- Pooled RMSE: 1.6375 minutes.
- Pooled R²: 0.9421.
- Macro fold mean ± SD RMSE: 1.4526 ± 0.7756 minutes.

| selected_model | fold_count |
|---|---|
| hist_gradient_boosting | 19 |
| random_forest | 1 |

### By held-out topology

| topology | fold_count | n_pooled_samples | pooled_mae | pooled_rmse | pooled_r2 |
|---|---|---|---|---|---|
| dense_mixed | 4 | 864 | 0.9695 | 1.5172 | 0.9034 |
| grid | 4 | 864 | 0.5477 | 0.9195 | 0.9233 |
| irregular_geometric | 4 | 864 | 1.1349 | 1.8930 | 0.9248 |
| radial | 4 | 864 | 0.3646 | 0.5808 | 0.9007 |
| sparse_mixed | 4 | 864 | 1.4481 | 2.5177 | 0.8869 |

## Conformal aggregate at alpha 0.10

| method | n_samples | nominal_coverage | empirical_coverage | mean_interval_width | macro_fold_std_coverage |
|---|---|---|---|---|---|
| absolute | 4320 | 0.9000 | 0.8965 | 4.5590 | 0.0815 |
| local | 4320 | 0.9000 | 0.9005 | 3.7221 | 0.0320 |
| relative | 4320 | 0.9000 | 0.8986 | 4.1398 | 0.0188 |

### By held-out topology

| topology | method | n_samples | empirical_coverage | mean_interval_width |
|---|---|---|---|---|
| dense_mixed | absolute | 864 | 0.8762 | 4.5813 |
| grid | absolute | 864 | 0.9653 | 4.5397 |
| irregular_geometric | absolute | 864 | 0.8565 | 4.7409 |
| radial | absolute | 864 | 0.9850 | 4.4254 |
| sparse_mixed | absolute | 864 | 0.7998 | 4.5077 |
| dense_mixed | local | 864 | 0.8808 | 3.8938 |
| grid | local | 864 | 0.9120 | 2.3951 |
| irregular_geometric | local | 864 | 0.9074 | 4.9418 |
| radial | local | 864 | 0.9178 | 1.6377 |
| sparse_mixed | local | 864 | 0.8843 | 5.7420 |
| dense_mixed | relative | 864 | 0.9005 | 4.7260 |
| grid | relative | 864 | 0.9005 | 2.5764 |
| irregular_geometric | relative | 864 | 0.9074 | 5.5017 |
| radial | relative | 864 | 0.8993 | 1.6464 |
| sparse_mixed | relative | 864 | 0.8854 | 6.2484 |

## Per-fold results

| fold_id | heldout_network_id | heldout_topology | selected_model | mae | rmse | r2 | absolute | relative | local |
|---|---|---|---|---|---|---|---|---|---|
| lono_18_dense_mixed_calibration | dense_mixed_calibration | dense_mixed | hist_gradient_boosting | 1.0929 | 1.7269 | 0.8896 | 0.8287 | 0.8843 | 0.8935 |
| lono_19_dense_mixed_test_id | dense_mixed_test_id | dense_mixed | hist_gradient_boosting | 1.1188 | 1.7082 | 0.9018 | 0.8241 | 0.8935 | 0.8472 |
| lono_16_dense_mixed_train | dense_mixed_train | dense_mixed | hist_gradient_boosting | 0.7742 | 1.2037 | 0.9174 | 0.9259 | 0.9352 | 0.9259 |
| lono_17_dense_mixed_validation | dense_mixed_validation | dense_mixed | hist_gradient_boosting | 0.8923 | 1.3631 | 0.8650 | 0.9259 | 0.8889 | 0.8565 |
| lono_02_grid_calibration | grid_calibration | grid | hist_gradient_boosting | 0.6196 | 1.0959 | 0.8861 | 0.9352 | 0.9259 | 0.9167 |
| lono_03_grid_test_id | grid_test_id | grid | hist_gradient_boosting | 0.6675 | 1.0152 | 0.9311 | 0.9398 | 0.9167 | 0.8935 |
| lono_00_grid_train | grid_train | grid | hist_gradient_boosting | 0.3493 | 0.5089 | 0.9006 | 1.0000 | 0.8796 | 0.9074 |
| lono_01_grid_validation | grid_validation | grid | hist_gradient_boosting | 0.5545 | 0.9440 | 0.8165 | 0.9861 | 0.8796 | 0.9306 |
| lono_08_irregular_geometric_ood_0 | irregular_geometric_ood_0 | irregular_geometric | hist_gradient_boosting | 1.0239 | 1.7202 | 0.9339 | 0.8704 | 0.9120 | 0.9444 |
| lono_09_irregular_geometric_ood_1 | irregular_geometric_ood_1 | irregular_geometric | random_forest | 1.1969 | 1.8699 | 0.9459 | 0.8750 | 0.9259 | 0.9491 |
| lono_10_irregular_geometric_ood_2 | irregular_geometric_ood_2 | irregular_geometric | hist_gradient_boosting | 1.2201 | 2.1068 | 0.9028 | 0.8380 | 0.8889 | 0.8750 |
| lono_11_irregular_geometric_ood_3 | irregular_geometric_ood_3 | irregular_geometric | hist_gradient_boosting | 1.0986 | 1.8548 | 0.8941 | 0.8426 | 0.9028 | 0.8611 |
| lono_06_radial_calibration | radial_calibration | radial | hist_gradient_boosting | 0.4110 | 0.6758 | 0.8892 | 0.9769 | 0.9167 | 0.9213 |
| lono_07_radial_test_id | radial_test_id | radial | hist_gradient_boosting | 0.4506 | 0.6760 | 0.8854 | 0.9676 | 0.8981 | 0.8935 |
| lono_04_radial_train | radial_train | radial | hist_gradient_boosting | 0.2709 | 0.4353 | 0.8955 | 0.9954 | 0.8981 | 0.9491 |
| lono_05_radial_validation | radial_validation | radial | hist_gradient_boosting | 0.3261 | 0.4960 | 0.8633 | 1.0000 | 0.8843 | 0.9074 |
| lono_14_sparse_mixed_calibration | sparse_mixed_calibration | sparse_mixed | hist_gradient_boosting | 2.0095 | 3.6557 | 0.8363 | 0.6991 | 0.8750 | 0.8611 |
| lono_15_sparse_mixed_test_id | sparse_mixed_test_id | sparse_mixed | hist_gradient_boosting | 1.2942 | 2.0255 | 0.8999 | 0.8009 | 0.9028 | 0.8657 |
| lono_12_sparse_mixed_train | sparse_mixed_train | sparse_mixed | hist_gradient_boosting | 1.2601 | 1.9145 | 0.9044 | 0.8333 | 0.8657 | 0.8889 |
| lono_13_sparse_mixed_validation | sparse_mixed_validation | sparse_mixed | hist_gradient_boosting | 1.2284 | 2.0550 | 0.9014 | 0.8657 | 0.8981 | 0.9213 |

## Explicit unfavorable folds

The following are the five folds with the worst minimum conformal-method coverage, breaking ties by higher RMSE. They are retained without exclusion or post-hoc retuning.

| fold_id | heldout_network_id | heldout_topology | selected_model | rmse | r2 | absolute_coverage | relative_coverage | local_coverage |
|---|---|---|---|---|---|---|---|---|
| lono_14_sparse_mixed_calibration | sparse_mixed_calibration | sparse_mixed | hist_gradient_boosting | 3.6557 | 0.8363 | 0.6991 | 0.8750 | 0.8611 |
| lono_15_sparse_mixed_test_id | sparse_mixed_test_id | sparse_mixed | hist_gradient_boosting | 2.0255 | 0.8999 | 0.8009 | 0.9028 | 0.8657 |
| lono_19_dense_mixed_test_id | dense_mixed_test_id | dense_mixed | hist_gradient_boosting | 1.7082 | 0.9018 | 0.8241 | 0.8935 | 0.8472 |
| lono_18_dense_mixed_calibration | dense_mixed_calibration | dense_mixed | hist_gradient_boosting | 1.7269 | 0.8896 | 0.8287 | 0.8843 | 0.8935 |
| lono_12_sparse_mixed_train | sparse_mixed_train | sparse_mixed | hist_gradient_boosting | 1.9145 | 0.9044 | 0.8333 | 0.8657 | 0.8889 |

## Scientific limitations

1. Twenty synthetic networks, only four per family, are a small corpus. These folds are strongly dependent because the same networks recur in fitting roles.
2. Leave-one-network-out performance on these generators is not universal network, city, traffic, medical, or operational generalization.
3. Original Step 9 networks vary in size, density, and base-time multiplier. Topology summaries therefore mix topology with finite configured instance variation.
4. Conformal guarantees are exchangeability-based; systematic network or topology shifts can cause under-coverage, which is reported rather than repaired using held-out data.
5. No held-out fold is used to choose a model, residual scale, conformal quantile, normalization, or alpha. Consequently, unfavorable held-out behavior is not adaptively corrected.
6. This experiment contains no quantum circuit or hardware execution and supports no speedup claim.

## Reproduction

Run `python experiments/run_step9_1_leave_one_network_out.py` from the repository root. The runner refuses to overwrite an existing result directory. The manifest records both configuration hashes, package versions, seeds, source hashes, and artifact hashes.
