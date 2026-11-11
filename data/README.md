# Eye Sentinel data workspace

**Current status (2026-11-10):** `prepared/formal_v3/` completed three 20-epoch models and all 13,789 teacher targets. EyeQ has 28,792 imported, decoded and hashed originals on D:, with cross-source association evidence under `audit/session_20261110/eyeq_overlap/`; it is not released for training. Two newly linked boundary duplicates are excluded by the separate evaluation-only `prepared/formal_v3_boundary_repaired_20261110/` subset (1,230 rows); use its generated boundary scores for further development. Original train/model_val/final tests and checkpoints remain unchanged. See [latest experiment audit](../docs/EXPERIMENT-AUDIT-1110.md). The following v2 preparation descriptions are historical.

**Current audit (2026-11-09):** [readiness report](../docs/AUDIT-1109.md), [training protocol](../docs/TRAINING-PROTOCOL-1109.md), and [Luna image review](audit/2026-11-09/luna/audit.md). New evidence is under `audit/2026-11-09/`; source files were not changed. Full decoding found one truncated FIQS original and ten exact duplicate pairs in HYGD. Candidate worklists are not frozen training sets. The earlier [proposal-1106](../proposal-1106.pdf), [registry](DATASETS-1106.md), [review](../docs/proposal-1106-review.md), and v4 descriptions below are historical. Original sources and audit records retain their paths.

**Historical v2 research training preparation:** `prepared/formal_v2/` has complete teacher targets and passed real-GPU training/restart checks for its historical configuration. See [current readiness](../docs/TRAINING-READINESS.md) and [runbook](../docs/TRAINING-RUNBOOK.md). Deep Luna audits are under `audit/deep/`. `prepared/formal/` is superseded; technical image-family quarantine in v2 prevents an annotated duplicate from re-entering through a second source. Unknown patient identity and source-specific label limits remain recorded.

Audit and reorganization date: 2026-10-22. Proposal v4 focuses on quality-aware glaucoma screening from fundus photographs. The workspace also retains Wide-Field and FrontofEyeDx for future extensions.

For source counts, grouping rules and the v4 experiments, see [Eye Sentinel Dataset Introduction](../Eye_Sentinel_Dataset_Introduction.pdf) and its [Markdown source](../Eye_Sentinel_Dataset_Introduction.md).

## Layout

- `sources/eyeworld/`: expanded EyeWorld source tree; original Fundoscopy and Wide-Field folders and ODIR files remain grouped by source.
- `sources/frontofeyedx/Dataset/`: FrontofEyeDx candidate images grouped by the original disease-folder labels.
- `metadata/eyeworld/`: the 32,178-row EyeWorld image index.
- `metadata/frontofeyedx/`: the disease worklist.
- `review/exact_duplicate_label_conflicts/`: one copy each of 22 identical images that appeared under different disease-folder labels. These images are excluded from the class folders pending label review.
- `audit/baseline/data_audit/`: the pre-cleanup inventory and report; its paths document the former `Eye Project Data` layout.
- `audit/current/`: the post-cleanup inventory, class counts, ODIR summaries, and scripts.
- `archive/macosx_sidecars/`: extracted `__MACOSX` AppleDouble sidecars, kept out of the image-source tree and excluded from model/audit counts.
- `audit/source_archive_validation.json` and `audit/archive_cleanup_action.json`: content verification and the later removal record for the redundant ZIP.

## What is in the sources

EyeWorld contains Fundoscopy and Wide-Field images from several heterogeneous collections. Before cleanup it had 32,178 image-path records and 32,150 distinct file hashes. FrontofEyeDx has 2,321 images across 94 populated disease folders; counts range from 1 to 98 per folder (median 23), and 68 folders contain fewer than 30 images. The combined sources contain 34,471 distinct image contents. The current source and review folders should contain one physical file per SHA-256 hash.

The EyeWorld workbook uses `id`, `filename`, `folder`, `dx`, `path`, `caption`, and `label`. The `caption`, path, folder name, and diagnosis fields can reveal the answer; do not pass them to an image model as input features. The workbook does not provide a complete per-image acquisition source, device, symptoms, or adjudicated clinical reference label.

ODIR is a subset with 7,000 eye-image rows and 3,500 `orig_id` values; each ID has a left and right eye record. Split this subset by `orig_id` to keep paired eyes together. The ODIR metadata does not cover the rest of EyeWorld.

## Cleanup record

- Removed 4,214 images from the extra root `Dataset` tree after every file hash matched an EyeWorld image. The root copy added no unique image content.
- Removed 28 extra exact-hash image files within EyeWorld. One physical image remains for each hash. All original path-to-label memberships are recorded in `audit/source_image_manifest_precleanup.csv`; review conflicts are listed in `audit/exact_hash_label_conflicts.csv`.
- Removed four byte-identical README/workbook copies; see `audit/metadata_dedup_actions.csv`.
- Verified the original 4,588,408,810-byte ZIP against the expanded EyeWorld tree, then removed the redundant compressed copy on explicit request. The verification and removal records remain in `audit/`.
- Exact-hash checks do not detect crops, resizing, recompression, panels from the same case, or other near-duplicates. No clinical label adjudication or full-pixel decode was performed.

## Use with proposal v4

The binary candidate pool contains 3,726 glaucoma and 3,998 normal fundus images. ACRIMA contributes 396 glaucoma and 309 normal images within that total. Hold out its full 705-image source for evaluation, leaving 7,019 development candidates. Review folder labels, visible diagnostic text and image suitability, then group augmentation families and related versions before splitting. ODIR is a separate extension with 3,500 patient pairs grouped by `orig_id`.

Proposal v4 uses MobileNetV3-Small, temperature scaling, retinal sharpness/contrast/exposure measures, and a combined quality-and-uncertainty review score. Compare random, uncertainty-only and combined ranking at 5%, 10% and 20% review allowances. Report released-negative glaucoma misses, NPV, negative coverage, review error capture and total referral workload. The dataset guide defines the retake workflow, source holdout, controlled degradations and CPU targets.
