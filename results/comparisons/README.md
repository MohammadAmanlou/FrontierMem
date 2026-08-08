# Cross-model comparison artifacts

Generated with `scripts/compare_qwen_models.py`.

Included:
- `all_model_metrics_raw.csv`: local baselines + Qwen 0.5B/1.5B/3B aggregate metrics.
- `all_model_metrics_percent.csv`: same metrics in percentage units.
- `qwen_3b_vs_1_5b_deltas.csv`: direct 3B minus 1.5B comparison for the shared `gold_memory_llm` mode.
- `all_prediction_distributions.csv`: prediction shares for all Qwen runs available in this snapshot.
- `all_collapse_analysis.csv`: dominant-action / collapse diagnostic.
- PNG figures for overall Qwen metrics, boundary metrics, and action distributions.

The 3B per-example prediction file itself was not part of the uploaded artifact set, so only its aggregate distribution and metrics are included. The exact executed 3B log is in the Kaggle experiment notebook.
