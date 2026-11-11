# Formal Training Readiness Implementation Plan

> **For agentic workers:** Continue inline using executing-plans; user explicitly authorizes multiple batches of Luna audit agents. Data auditors own separate audit directories; the primary agent owns training code and integration.

**Goal:** Produce frozen research training manifests, runnable glaucoma/quality training jobs, a complete allowed-data teacher cache, and real-GPU forward/backward/save/resume evidence.

**Architecture:** Keep raw sources and upstream model code immutable. Merge patient, exact-content and conservatively confirmed image-family edges into split units. Save explicit eligibility/provenance and separate source-specific endpoints. Train independent disease and quality models; derive p–q review boundaries only after validation predictions exist.

**Tech Stack:** Existing project Python3.11/CUDA environment, PyTorch/torchvision, PIL, NumPy/SciPy/OpenCV, CSV/JSON.

**Spec:** `../../TRAINING-PROTOCOL-1109.md` plus user authorization to finish formal training preparation.

## Global Constraints

- No raw images or labels are deleted or overwritten.
- Neither test labels nor test inputs enter training, teacher-cache training targets, model selection or boundary fitting.
- Unknown patient identity stays unknown; image-family isolation cannot be described as verified patient-level independence.
- Source glaucoma/normal, RG/NRG and GON labels remain separate endpoint metadata; no pooled clinical claim across different definitions.
- Luna technical observations are not clinical label adjudication.
- FIQS teacher exposure prevents claiming an independent teacher quality test; student fidelity validation is labeled as such.

## Review Focus

- Transitive patient/duplicate families must never span subsets or model tasks.
- Test paths and unresolved exclusion candidates must be rejected by training/cache commands.
- Checkpoints must bind manifest/config/pretrained hashes and restore RNG, optimizer and batch position.
- Quality labels stay attached to image identity and teacher/preprocess versions; image-altering augmentation cannot silently retain an incompatible cached quality target.
- No empty, single-class or unbounded/nonfinite data silently passes readiness.

## Task 1: Parallel deep audit and frozen cohort

Files: `data/audit/deep/luna_{provenance,visual,duplicates}/`, `tools/freeze_training_cohort.py`, `tests/test_cohort.py`, `data/prepared/formal/`.

Interfaces: frozen manifest rows contain sample_id,path,sha256,source,target_definition,label,related_group,patient_id_status,split,task,eligible,review_status. Release JSON records hashes and audit decisions. Input confirmed edges and explicit visual exclusions are traceable to audit evidence.

- [x] Complete multiple Luna audit batches, review high-risk pairs and label/group rules.
- [x] Test group-transitivity, holdout precedence, conflicting labels, unknown identities and task/test overlap rejection; observe failure before implementation.
- [x] Generate source-aware frozen splits and exclusion/alias files with seed and limitations; verify content hashes and group disjointness.

## Task 2: Guarded teacher cache and training commands

Files: `training/{data,models,teacher,engine}.py`, `tools/{cache_teacher,train_model}.py`, `configs/`, `tests/test_training.py`.

Interfaces: `train_model.py --config FILE [--resume CHECKPOINT] [--max-steps N]`; configs bind frozen manifests and a run directory. Disease loss is binary BCE; quality distillation is continuous Huber, preserving raw teacher scores. Eval exports identity,prediction,source,group,truth without selecting gray boundaries. Teacher cache only permits frozen quality train/validation images, rejects disease/test overlap, and fingerprints source/model/preprocessing.

- [x] Add tests for manifest leakage/eligibility, quality target identity, deterministic sampling and safe checkpoint compatibility/resumption.
- [x] Implement minimal independent EfficientNet-B0 disease/quality models and ResNet50 comparator candidate with verified ImageNet initialization.
- [x] Implement deterministic training order, image transforms, optimizer/AMP, validation, best/last checkpoint and exact resume state.
- [x] Generate full allowed teacher cache; preserve failed/nonfinite cases explicitly.

## Task 3: Real-data readiness checks and audit integration

Files: `runs/preflight/`, `docs/{TRAINING-RUNBOOK,TRAINING-READINESS}.md`, current README/protocol/status.

- [x] Real GPU batches for disease A, quality student and comparator candidate; finite loss/gradients, parameter change and checkpoint restore.
- [x] Verify uninterrupted vs checkpoint-resumed next step within declared tolerance; do not use smoke checkpoints as trained research results.
- [x] Fresh independent review; fix meaningful issues and run relevant tests.
- [x] Publish exact training commands, cohort counts, environment/config/model/data hashes and honest remaining scientific limits. Mark ready only when all required preparation checks pass.

## Execution ledger

- Existing project is not a Git repository; code lives in new training/tests/configs directories, sources remain in place.
- Prior preflight succeeded: original teacher CUDA forward, 20 crops, SoftPool formula check, official EfficientNet initialization.
- Batch 1 Luna tasks launched: provenance/grouping, extended visual review, conservative near-duplicate verification. Root proceeds with code while audits run.
- Completed further Luna batches for comparator/source protocol, targeted technical decisions, two-pass code review and independent frozen release review.
- Ruling: keep unknown patient identity explicit and use a research-only image-family-separated release, with whole HYGD/ACRIMA external cohorts; no all-patient-independence claim. EyeAugmented is excluded until parent lineage is available.
- Ruling: treat all FIQS-related disease components as possible public-teacher exposure; keep them in train/val and quarantine connected external candidates. This is source-based protection before model results.
- Corrected a real v1 issue: an annotated ODIR photo's Kaggle alias survived path-only quarantine. Added red→green family-level regression test, froze formal_v2, retained and blocked v1. Exactly one Kaggle training row changed; all retained rows and external cohorts are identical.
- Teacher cache: 2,246 finite 20-crop targets computed; v2 quality identities/splits/content and teacher implementation are identical, so targets were explicitly verified and rebound with saved lineage. SQLite transactional recovery and identity guards tested.
- Real-GPU evidence: all three tasks update parameters, produce validation predictions/best/last checkpoints, and reproduce the next optimizer step exactly after restart (weight/loss differences 0.0).
- Final tests: 31 passed. Independent formal_v2 audit: PASS_WITH_LIMITATIONS, issues empty. Machine gate: data/prepared/formal_v2/readiness.json research_training_ready=true. No formal 20epoch training started.
