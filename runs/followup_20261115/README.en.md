# November 15 three-seed training results

[中文](README.md) · [Full results analysis](../../docs/progress/2026-11-15/completed-experiments/completed-results.en.md)

The cohort is frozen release `expanded_20261113`. All 18 runs have completed training and development evaluation: six conditions with three common seeds, each trained for 2,400 parameter updates, saving checkpoints and model-validation predictions every 300 updates. See the [experiment plan](experiment_plan.json).

- [Driver status](pipeline/status.json): current run, runs with completed development evaluation, and failure information.
- [CUDA recovery preflight](preflight/verification.json): uninterrupted and resumed weights, step records, and state match exactly.
- [Available results](comparison.json): source-specific results by update, refreshed after each completed run.

Start or resume from the project root with `& ./tools/start_followup_experiments_20261115.ps1`. The background driver trains and evaluates sequentially, skips completed training budgets, and resumes interrupted runs from `last.pt`. Changed input fingerprints reject recovery; changes to training implementation during an active run require a new experiment version.

Local `seed<seed>/<condition>/validation_checkpoints/` holds full weights and validation predictions. `development/update_<count>/` holds results for original development photographs and the fixed synthetic stress subset. `eyeq_development/` stores alternative quality inference and `error_ranking/` stores earlier image joins and grouped cross-validation scores. Final tests have not been scored.

`completed_analysis/` holds new review scores for the 18 selected models. The report's [completed-analysis.json](../../docs/progress/2026-11-15/completed-experiments/completed-analysis.json) retains seed summaries, case intervals, quality strata, actual sampling exposure, and input hashes.
