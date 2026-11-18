# Training runbook

Updated November 15, 2026 · [中文](training-runbook.md)

`followup_20261115` uses frozen data release `expanded_20261113`. All 18 runs have completed training and development evaluation: six conditions with three seeds, each trained for 2,400 parameter updates. See the [full training and quality analysis](../progress/2026-11-15/completed-experiments/completed-results.en.md). The previous four 1,200-update runs are also complete. Run commands from the project root, `team_project/`.

## Completed-training status and reproduction

```powershell
Get-Content runs/followup_20261115/pipeline/status.json
& ./tools/start_followup_experiments_20261115.ps1
```

The launcher checks for an existing driver, then trains and evaluates sequentially in a hidden background process. Completed training budgets are skipped; interrupted runs resume from `last.pt`. Input fingerprints bind data, configuration, initialization, and training implementation.

`configs/followup_20261115/seed<seed>/` holds six condition configurations. `runs/followup_20261115/seed<seed>/<condition>/validation_checkpoints/` saves full weights, predictions, and metrics every 300 updates. Selection uses five main sources' mean AUROC. Development evaluation retains update-1,200, update-2,400, and selected-checkpoint results. `runs/followup_20261115/comparison.json` updates after each evaluated run, and `preflight/verification.json` records the three CUDA recovery checks.

```powershell
tools/training_python.cmd tools/evaluate_followup_20261115.py --summarize
python tools/analyze_completed_followup_20261115.py
python tools/report_completed_followup_20261115.py
python tools/verify_completed_followup_20261115.py
```

These commands analyze saved predictions and build bilingual reports, PDFs, LaTeX, and ten figure sets in `completed-experiments/`. Image-level review scores are saved under `runs/followup_20261115/completed_analysis/`. Earlier reports at the date root retain the training-launch snapshot.

## November 13 experiment records

## Environment, configurations, and outputs

`tools/training_python.cmd` uses the project's `.envs/training` Python and CUDA environment. Document figures and PDFs use default Python, Matplotlib, Pandoc, XeLaTeX, and Poppler.

| Location | Purpose |
|---|---|
| `data/prepared/expanded_20261113/` | Joint frozen manifest, release information, and allocation checks. |
| `configs/expanded_20261113/` | Original-data, expanded-clean, 30%, and 50% degradation configurations. |
| `runs/expanded_20261113/<arm>/` | Weights, logs, validation and development predictions, and summaries. |
| `runs/expanded_20261113/pipeline/status.json` | Experiment execution status. |
| `docs/progress/2026-11-13/` | Bilingual reports, LaTeX, PDFs, figures, and report verification. |

The arms share EfficientNet-B0 initialization, seed 20261113, 320-pixel inputs, a maximum batch size of 16, AdamW learning rate 0.0001, and 1,200 optimizer updates. Common model validation runs every 300 updates. `best.pt` maximizes the configured mean source AUROC; `last.pt` holds the budget-end checkpoint. These are updates 300 and 1,200, respectively, for expanded data without degradation.

## Check completion

```powershell
Get-Content runs/expanded_20261113/pipeline/status.json
tools/training_python.cmd tools/verify_expansion_delivery_20261113.py
python tools/verify_expansion_analysis_20261113.py
```

Full verification requires local datasets, weights, and image-level predictions. Repository result JSON, figures, and reports support inspection of summary values. Checks cover manifest hashes, update counts, common development identities, statistical results, and report links.

## Execution and recovery

The common entry point is:

```powershell
& ./tools/start_expansion_experiments_20261113.ps1
```

It runs in a hidden background window and writes logs under `runs/expanded_20261113/pipeline/`. The driver checks GPU recovery evidence and each arm's completion. Completed arms skip training updates and repeat development evaluation. Interrupted arms resume from `last.pt`. Budget-end development evaluation is generated separately by `tools/run_budget_end_review_20261113.py`.

Checkpoints bind configuration, initialization, data manifests, and training-code hashes, which are checked on recovery. Changes to sampling, augmentation, or budget use a new configuration and output directory, with matching preflight checks.

## Build reports

```powershell
python docs/progress/2026-11-13/data-review/build_pdfs.py
python tools/report_expansion_training_20261113.py
python tools/report_expansion_analysis_20261113.py
python tools/build_progress_overview_20261113.py
python tools/verify_documentation_20261113.py
```

Report generators update their own reports and figures. Maintain progress indexes and navigation separately. Before delivery, check bilingual consistency, figure units and sample sizes, every PDF page, and retained LaTeX sources.

The [historical runbook](../historical/2026-11-09/training-runbook.md) retains `formal_v2` operations. The [November 12 report](../progress/2026-11-12/training-progress.en.md) covers `formal_v3` and EyeQ training.
