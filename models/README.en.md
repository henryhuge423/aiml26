# Model resources

English | [中文](README.md) · Updated November 13, 2026

The project uses EfficientNet-B0 disease classifiers, a ResNet50 comparator, and an EfficientNet-B0 quality student. The quality teacher uses the author's FTHNet implementation and weights. `checkpoints/` holds initialization and teacher files; `third_party/` holds upstream code. Trained weights reside in their corresponding `runs/` directories.

Earlier disease and quality models completed 20 epochs. This round's four `expanded_20261113` EfficientNet-B0 arms each completed 1,200 updates, retaining validation-selected and budget-end checkpoints. The [latest overview](../docs/progress/2026-11-13/progress-overview.en.md) and [training report](../docs/progress/2026-11-13/training-results/training-results.en.md) identify results and evaluation sets.

Full SHA256 hashes of initialization, teacher, and trained weights are recorded in model registries, configuration fingerprints, and run artifacts. Weights stay local; the repository shares implementations, configurations, and selected results. The [runbook](../docs/operations/training-runbook.en.md) provides environment and usage details. The [historical preparation record](../docs/historical/2026-11-09/model-preparation.md) retains the original model and teacher setup.
