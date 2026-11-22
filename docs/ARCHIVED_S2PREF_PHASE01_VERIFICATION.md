# Verified historical S2Pref Phase-01 scores — 2026-11-22

Audited from the previously produced user archive `FrontierMem_Phase_01.zip`:
`phase_01/context_flip_metrics_old_models.csv`,
`phase_01/metrics_old_models_s2pref.csv`,
`phase_01/phase_status.json`, and `phase_01/s2pref_schema_audit.json`.
The full underlying archive is **not** republished here.

| Metric | Historical Pointwise | Historical CAID |
|---|---:|---:|
| Derived-silver test decisions | 400 | 400 |
| Context pairs (20 source-record groups) | 200 | 200 |
| Binary accuracy | 60.0% | 60.0% |
| Binary macro-F1 | 57.4457% | 58.1546% |
| APPLY recall | 35.5% | 39.0% |
| Wrong APPLY on IGNORE | 15.5% | 19.0% |
| Pair direction | 80.0% | 86.0% |
| Strict flip | 25.0% | 21.5% |

Upstream sample construction: 128 source records read from a stream prefix; 640 aspects; 2,560 derived silver atoms; heldout test 20 source groups, 200 pairs, 200 APPLY plus 200 IGNORE decisions.

**Limits:** derived silver labels are NOT native gold RPEval labels or the complete S2Pref task. Existing Qwen checkpoints are not fully training-budget matched. Pair-direction gain without strict-flip gain is a motivation for context boundary learning, not proof of novelty or superiority. Inference results were archived historically; this file is not a new GPU run. Subsequent bootstrap resampling must use source groups, not treat every pair as independent.

**Protocol:** preserve the exact 20 held-out source group IDs; build any Dual-Axis training data from separate source records only. Official RPEval tests and BenchPreS are never used to optimize memory selection or thresholds.
