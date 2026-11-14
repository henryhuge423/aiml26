# November 13, 2026: new datasets and degradation experiments

English | [中文](README.md)

The day's work combines data preparation, degradation-method checks, four training arms, and result analysis. Read the overview first, then follow the sequence below for methods and evidence.

| Reading order | Markdown | PDF | Contents |
|---|---|---|---|
| Overview | [Progress overview](progress-overview.en.md) | [PDF](progress-overview.en.pdf) | Completion status, main results, dataset sizes, and next priorities. |
| 1. Data and methods | [Data and degradation review](data-review/data-and-degradation-review.en.md) | [PDF](data-review/data-and-degradation-review.en.pdf) | 45 public sources, 9,070 added images, labels and identity links, degradation research, and evaluation design. |
| 2. Training and evaluation | [Training results](training-results/training-results.en.md) | [PDF](training-results/training-results.en.pdf) | Joint release, four 1,200-update arms, original-image and quality evaluation; supporting [data audit](training-results/data-audit.en.md) and [degradation audit](training-results/degradation-audit.en.md). |
| 3. Interpretation | [Statistical analysis](experimental-analysis/experimental-analysis.en.md) | [PDF](experimental-analysis/experimental-analysis.en.pdf) | Paired source differences, sampling counts, checkpoint-selection effects, defocus, and real-quality cases. |

The detailed reports describe successive stages of the same day. Images were grouped and allocated to training, validation, development, and test roles after the additive-pool review. Main comparisons in the overview and analysis use update-1,200 checkpoints; the training report also records each arm's validation-selected checkpoint.

Each main report retains matching Chinese and English `.md`, `.pdf`, and `.tex` files. Figures, summary data, and verification records accompany each topic. The [analysis JSON](experimental-analysis/analysis.json) and [source-contrast CSV](experimental-analysis/paired-source-contrasts.csv) support numerical checks.

[All progress reports](../README.en.md) · [Documentation guide](../../README.en.md) · [Next steps](../../operations/next-steps.en.md)
