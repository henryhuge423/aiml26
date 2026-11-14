# Expanded-data training record

English | [中文](README.md)

Each of four arms completed 1,200 optimizer updates, with common model validation every 300 updates. `best.pt` maximizes mean AUROC over 11 sources; `last.pt` is update 1,200. Expanded data without degradation selects update 300; the other arms select update 1,200. Original and fixed synthetic-stress development results are retained separately.

[Progress overview](../../docs/progress/2026-11-13/progress-overview.en.md) · [Training report](../../docs/progress/2026-11-13/training-results/training-results.en.md) · [Statistical analysis](../../docs/progress/2026-11-13/experimental-analysis/experimental-analysis.en.md)

`comparison.json` evaluates selected checkpoints; `budget_end_comparison.json` evaluates the same update 1,200 across arms. Each arm's `run_summary.json` records updates and sampling counts. `development_metrics.json` and `last_development_metrics.json` record metrics for these respective checkpoints.

[Four configurations](../../configs/expanded_20261113/) · [GPU recovery preflight](preflight/verification.json) · [Runbook](../../docs/operations/training-runbook.en.md)

Weights, image-level predictions, and full logs stay local; these summaries are saved with the repository. The next round uses new configurations and output directories.
