# Data workspace

English | [中文](README.md) · Updated November 13, 2026

The current disease-training and evaluation release is `prepared/expanded_20261113/`: 24,491 images comprising 15,480 training, 2,046 model validation, 2,053 rule development, 3,470 internal/added reserved tests, 705 ACRIMA external tests, and 737 HYGD external tests. See the [joint release guide](prepared/expanded_20261113/README.en.md), [data audit](../docs/progress/2026-11-13/training-results/data-audit.en.md), and [progress overview](../docs/progress/2026-11-13/progress-overview.en.md).

The additive pool, `prepared/expansion_20261113/`, contains 9,070 images with explicit binary labels and a separate 204-image suspect pool. It records the source-review stage; the joint release records subsequent allocation to training, validation, development, and tests. Manifests and audits retain file hashes, RGB-pixel hashes, source labels, known patients, and confirmed image links.

| Directory | Contents |
|---|---|
| `sources/`, `external/`, `external_glaucoma/`, `external_quality/` | Original source files and subsequent downloads. |
| `metadata/` | Source indexes and label tables. |
| `prepared/` | Versioned frozen manifests, releases, and grouping checks. |
| `audit/` | Source, identity, label, and allocation checks. |
| `review/` | Images for review and identical-image label conflicts. |
| `archive/` | Source packages and historical auxiliary files. |

Earlier `formal_v3` disease models and the original quality model completed 20 epochs, as did the EyeQ quality model. Quality data include FIQS and EyeQ. EyeQ import covers 28,792 originals, with eligible subsets used for quality training. The joint disease release allocates real poor-quality positives as 72 training, 12 validation, 12 rule-development, and 50 test images. FIVES and MSHF retain separate quality definitions.

Datasets, image-level manifests, and full audits retain their local paths. The repository shares source guides, bilingual reports, frozen-release summaries, and selected result JSON. The [historical data layout](../docs/historical/2026-10-22/data-workspace.md) and [early dataset registry](DATASETS-1106.md) preserve earlier records.
