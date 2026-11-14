# Expanded disease-training release

November 13, 2026. The frozen release contains 24,491 disease images: 15,480 training images, 2,046 model-validation images, 2,053 rule-development images, 3,470 internal/new held-out test images, 705 ACRIMA external-test images, and 737 HYGD external-test images.

The additions comprise 5,142 training, 858 validation, 833 rule-development, and 2,237 held-out images. The corrected original training portion contains 10,338 images. Use `origin_cohort` to select `formal_v3` or `expansion_20261113` training rows while retaining the same validation set across ablations. `imaging_domain` identifies KEH optic-disc crops.

Known patients and confirmed photograph groups remain within one role. PAPILA is a whole-source holdout. FIVES and Fundus-AVSeg official tests remain in test. All original disease tests are retained. Eighteen original training/validation photographs linked to official EyeQ tests are excluded from this release, together with twelve previously repaired rule-development images.

The release contains 72 real poor-quality training positives, 12 model-validation positives, 12 rule-development positives, and 50 test positives. FIVES uses `quality_fives_any_poor_dimension=True`; MSHF uses `quality_overall_official=0`. The definitions are reported separately. The 204 auxiliary suspects remain outside binary training.

All file hashes and full RGB decodes passed. Added images match the audited original RGB hashes. Existing quality models remain frozen; this is a disease-only release.

[Data audit](../../../docs/progress/2026-11-13/training-results/data-audit.en.md) · [中文](README.md)
