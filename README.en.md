# Eye Sentinel

English | [中文](README.md)

Eye Sentinel studies how glaucoma-classification scores and fundus-image quality scores can select photographs for manual review. The project will compare quality-guided review with a second disease model selecting the same number of images by uncertainty, measuring errors and glaucoma misses among the remaining automated results.

## Current progress · November 18, 2026

The matched ResNet-50 completed 2,400 updates. The [matched backbone comparison](docs/progress/2026-11-18/backbone-comparison.en.md) reports misses, false positives, and paired intervals for the two architectures using the same development photographs, validation calibration cohorts, and review workload. [English PDF](docs/progress/2026-11-18/backbone-comparison.en.pdf) · [中文 PDF](docs/progress/2026-11-18/backbone-comparison.zh.pdf)

At about 10% matched review, adding whole-image quality to the same disease classifier left three fewer misses on pooled development data and three more with AIROGS-only calibration. The gray-zone figures, workload comparison, and reserved-cohort inventory are complete. Two candidate cutoffs are fitted, and independent testing awaits the primary cohort definition. See the [gray-zone evaluation and proposal review](docs/progress/2026-11-18/gray-zone.en.md) ([English PDF](docs/progress/2026-11-18/gray-zone.en.pdf) · [中文 PDF](docs/progress/2026-11-18/gray-zone.zh.pdf)).

## Three-seed training · November 15, 2026

All 18 runs are complete: six conditions with three common seeds, each trained for 2,400 parameter updates, followed by development evaluation. Expansion improves LAG and KEH in all seeds. Restoring original-source sampling provides small AIROGS gains, while ODIR changes with training stage. Degradation mainly improves synthetic defocus. Across the 18 selected models, adding quality scores selects between four fewer and four more errors at the same 411-image review workload. Real poor-quality training photographs comprise 72 positives and three negatives.

See the [completed three-seed analysis](docs/progress/2026-11-15/completed-experiments/README.en.md) ([English PDF](docs/progress/2026-11-15/completed-experiments/completed-results.en.pdf) · [中文 PDF](docs/progress/2026-11-15/completed-experiments/completed-results.zh.pdf)), with ten bilingual figures and separate selected-checkpoint and update-2,400 results. [pipeline/status.json](runs/followup_20261115/pipeline/status.json) records completion; the [dated index](docs/progress/2026-11-15/README.en.md) groups the day's materials.

## Previous experiments: November 13, 2026

This round completed public-data review, identity grouping, dataset allocation, and four EfficientNet-B0 experiments. The additive pool contains 9,070 images. The joint release includes 15,480 training images, 2,046 model-validation images, and 2,053 rule-development images. Each arm completed 1,200 optimizer updates, comparing original data, expanded data, and 30%/50% mixed degradation.

On the same rule-development originals at update 1,200, expansion raises KEH [AUROC](docs/progress/2026-11-13/progress-overview.en.md#observed-results) from 0.6627 to 0.9333 and LAG from 0.9465 to 0.9890, while AIROGS and ODIR decline. Under synthetic defocus, 50% degradation improves ranking and reduces false positives while increasing misses at threshold 0.5. Detection among 12 real poor-quality development positives improves from 7/12 to 12/12, with identical counts across the three expanded arms.

The next round addresses original-source sampling, checkpoint selection, and real poor-quality negatives. Classification thresholds, manual-review rules, and final testing remain to be completed. Both earlier disease models, the original quality model, and the EyeQ quality model completed 20 epochs, documented in the November 12 report.

See the [bilingual progress overview](docs/progress/2026-11-13/README.en.md) ([English PDF](docs/progress/2026-11-13/progress-overview.en.pdf) · [中文 PDF](docs/progress/2026-11-13/progress-overview.zh.pdf)) for results, sample context, and next steps. The day's data review, training report, and statistical analysis are now grouped under one date.

## Documentation and operations

- [Documentation guide](docs/README.en.md): current operations, research materials, and historical records.
- [Progress report index](docs/progress/README.en.md): bilingual reports, PDFs, and figures by date.
- [Current runbook](docs/operations/training-runbook.en.md) and [next steps](docs/operations/next-steps.en.md).
- [Project proposal](proposal-1106.pdf) and [full directory guide](docs/DIRECTORY.en.md).

## Data and implementation

Original disease data come from AIROGS-Light, ODIR-5K, and Kaggle Eye Diseases. The additive pool retains labels and groups from LAG, GRAPE, KEH, BEH, PAPILA, FIVES, MSHF, Fundus-AVSeg, LES-AV, HRF, CRFO, Chákṣu, and three original AIROGS referral-positive images. Quality models use FIQS and EyeQ. Manifests separately record source label definitions, imaging domains, and confirmed image links.

`configs/` and `training/` contain configurations and training code; `tools/` and `tests/` contain processing tools and automated checks. `data/` stores manifests and audits, while `models/` and `runs/` hold local weights and run artifacts. The repository shares code, documents, figures, and selected result summaries; raw datasets, weights, and full logs stay local.

[Data workspace](data/README.md) · [Joint data release](data/prepared/expanded_20261113/README.en.md) · [Additive pool](data/prepared/expansion_20261113/README.en.md)
