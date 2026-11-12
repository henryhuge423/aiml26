# Eye Sentinel

English | [中文](README.md)

Eye Sentinel studies how glaucoma screening scores and fundus image quality can guide referrals for human review. The project uses two disease classifiers and an image-quality model to compare review strategies at the same review volume.

## Current progress

As of November 11, 2026, both disease models, the original quality model, and a quality model trained with EyeQ labels have each completed 20 epochs. The disease models used 10,354 training images and 1,190 model-validation images. Disease and quality scores have been generated for a corrected development set of 1,230 images to support review-rule design.

The disease models achieved AUROCs of 0.9766 and 0.9831 on AIROGS validation images. The EyeQ-trained quality model achieved a three-grade macro-F1 of 0.811 on 117 EyeQ validation images. Classification thresholds, the review rule, and final testing are the next steps.

In a paired experiment, two levels of blur and two levels of darkening were applied to each of 179 development images. At a display threshold of 0.5, 8 of 83 images labeled positive scored below the threshold in their original form, compared with 27 after stronger blur. The reports describe the datasets, results, and figures in [English](docs/progress/eye_sentinel_full_progress_20261111_en.pdf) and [中文](docs/progress/eye_sentinel_full_progress_20261111.pdf).

## Data

The disease models use images from AIROGS-Light, ODIR-5K, and Kaggle Eye Diseases. The quality models use those images alongside FIQS and EyeQ. Data expansion also includes original AIROGS images, Chaksu, PAPILA, FIVES, MSHF, RIM-ONE, BEH, Real Fundus, and HRF. The [data expansion report](docs/degradation-and-data-2026-11-11.md) and [dataset inventory](data/DATASETS-1106.md) document label definitions, image associations, and the current role of each source.

## Project documents

- [Current proposal](proposal-1106.pdf)
- [Training status and dataset sizes](docs/TRAINING-STATUS-1110.md)
- [Experiment review and EyeQ processing](docs/EXPERIMENT-AUDIT-1110.md)
- [Training protocol](docs/TRAINING-PROTOCOL-1109.md) and [runbook](docs/TRAINING-RUNBOOK.md)
- [Data expansion and degradation experiments](docs/degradation-and-data-2026-11-11.md)
- [External glaucoma dataset audit](data/audit/data_expansion_20261111/batch_glaucoma_external.md) and [image-quality dataset audit](data/audit/data_expansion_20261111/batch_quality_external.md)
- [EyeQ quality model training](runs/eyeq_quality_v1/README.md)

The detailed experiment records are currently in Chinese.

## Directory guide

- `configs/` and `training/` contain experiment configurations and training code.
- `data/` contains datasets, manifests, and audit records.
- `docs/` contains progress reports, experiment records, and protocols.
- `models/` and `runs/` contain model resources and experiment outputs.
- `tools/` and `tests/` contain data-processing and training tools and tests.
- `notes/`, `references/`, and `archive/` contain background materials and earlier proposals.
