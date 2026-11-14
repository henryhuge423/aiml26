# Training Preflight Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. User has asked to continue the authorized training preparation; proceed inline.

**Goal:** Establish a verified teacher runtime and reproducible, non-destructive preparation of the audited data before formal training.

**Architecture:** Keep original datasets and vendor code unchanged. Install an isolated CUDA environment under `.envs/training`, exercise the original FTHNet with its actual checkpoint, and emit explicit manifests for confirmed technical fixes and unresolved review items. Formal training remains gated on independent splits and label review.

**Tech Stack:** Python 3.11, PyTorch 2.5.1 CUDA 12.4, torchvision 0.20.1, upstream SoftPool, PIL, CSV/JSON.

**Spec:** `../../TRAINING-PROTOCOL-1109.md`.

## Global Constraints

- No source images or labels are deleted or moved.
- No final-test images are used to train the quality student or choose the gray boundary.
- The gray boundary creates the actual review count; the comparison model reviews the same k images by uncertainty rank.
- Teacher predictions are not human image-quality truth.
- No model accuracy or clinical conclusions are fabricated from smoke tests.

## Review Focus

- CUDA runtime must actually execute a tensor operation, not just detect a GPU.
- Vendor SoftPool CUDA must execute before the checkpoint forward can be claimed verified.
- A truncated FIQS original must never enter the default quality manifest when a healthy 1024 version is selected.
- HYGD exact duplicates across patient IDs must be one related split group and not double-counted.
- Training manifests must retain unresolved gradability/provenance blockers and must not silently become approved splits.

## Task 1: Runtime and teacher verification

Files: `.envs/training/`, `tools/smoke_fthnet.py`, `data/audit/2026-11-09/teacher_runtime.json`.

Interface: `smoke_fthnet.py --image PATH --output PATH` loads the pinned vendor model and actual checkpoint, performs original preprocessing with a fixed seed, and records finite scalar output and runtime versions. This is a smoke result, not an accuracy evaluation.

- [x] Create isolated CUDA environment and record explicit package versions; verify GPU tensor execution.
- [x] Build upstream SoftPool with available compiler/toolkit; verify a real pooling operation, recording failures if blocked.
- [x] Verify strict checkpoint loading and original TargetNet forward on a real permitted sample. Save provenance and preprocessing settings.

## Task 2: Technical-cleaning manifests

Files: `tools/prepare_clean_manifests.py`, `tests/test_clean_manifests.py`, `data/prepared/1109/`.

Interface: `build_manifests(data_root: Path, audit_root: Path, output: Path)` selects one healthy 1024 image per FIQS ID and one canonical file per HYGD hash; writes exclusions/aliases and preserves label and related-group metadata. No train/test assignment is produced.

- [x] Add tests for duplicate-across-patient grouping, truncated-original/healthy-resized selection, and preservation of unresolved eligibility.
- [x] Run tests and observe expected failure, implement the smallest manifest builder, then rerun tests.
- [x] Run on actual audited data and verify selected images exist, audited decode status, and unique IDs/hashes have correct counts. Current source hashes are rechecked before selecting them.

## Task 3: Refresh readiness evidence

Files: `docs/historical/2026-11-09/project-audit.md`, `docs/operations/next-steps.md`, `models/README.md`, `data/audit/2026-11-09/model_registry.json`.

- [x] Record observed runtime outcomes and technical cleaning outputs without overwriting historical audit facts.
- [x] Report completed checks, unresolved near-duplicate/patient/label decisions, and next formal-training gates.

## Execution ledger

- Workspace has no Git repository; work directly in the existing authorized project, with new implementation files separated from sources/vendor code. No git/worktree/commit operations are applicable.
- Existing data audit is complete; do not repeat the full six-minute inventory without a reason.
- CUDA dependency dry-run succeeded; implementation should verify the actual installed environment.
- Runtime outcome: CUDA tensor operation, upstream SoftPool formula check, strict teacher checkpoint load, and real-image 20-crop forward all passed. No performance evaluation or training is implied.
- Ruling: pin setuptools75.1.0 to restore torch2.5's MSVC compiler hook; leave original SoftPool sources unchanged. Build uses CUDA12.6 versus torch's CUDA12.4; formula and teacher tests verify this local build only.
- Review: fresh reviewer found no Critical/Important issues. The minor source-change concern was addressed with a red→green hash-mismatch regression test. Four tests pass; environment text refreshed.
