# Expanded cohort training launch implementation plan

> Execute inline with superpowers:executing-plans; final fresh-context review before launch.

**Goal:** Launch and supervise actual training on the latest audited split and expanded teacher cohort.

**Architecture:** Preserve formal_v2 and the proposal. Freeze formal_v3 with provenance, rebind the identical 2,246 existing teacher targets and compute 11,543 new targets. Use batch16 and configurable nonpersistent workers; rebuilding loaders each epoch preserves the existing deterministic per-image/epoch augmentation. Run GPU jobs sequentially with resumable checkpoints and persisted pipeline status.

**Tech stack:** Existing Python3.11/PyTorch2.5.1 CUDA environment, SQLite, Windows multiprocessing.

**Spec:** ../specs/2026-11-09-data-expansion-design.md

## Constraints and review focus

- Preserve upstream AIROGS test770 and included validation769; no final-test or boundary targets in quality fitting.
- Keep source labels and known related-group isolation; retain unknown-patient limitations.
- EyeQ has no training originals: exclude it, defer ordinal supervision until actual labeled images exist. No fictitious ordinal targets or disease labels as MOS.
- Initialize from pinned ImageNet weights; never from resource/preflight weights.
- Resume must reproduce next optimizer step with new loader/batch settings, including epoch changes and CPU/CUDA RNG.
- A/B share data, batch16, learning rate and 20epoch budget. No test evaluation or gray boundary selection in this run.

## Tasks

- [x] 1. Verify runtime and freeze formal_v3 from proposal with source fingerprints, counts and all source-file hash/decode checks.
- [x] 2. Tests first: extend verified cache reuse to an identity-matched subset; allow configurable workers. Test actual worker output/epoch determinism and image hash failures. Implement minimum changes and run suite.
- [ ] 3. Rebind old teacher targets, generate new ones, verify all identities/finite scores and protected-role isolation.
- [ ] 4. New configs + real GPU continuous/resume/validation verification for A, B and quality, using >1 batches at batch16. Fresh review of release/code/evidence.
- [x] 5. Start sequential durable training pipeline, inspect actual epoch/checkpoint/metrics progress, persist launch/status and update entry docs + automation memory.

## Execution ledger

- No git repository exists in this workspace; isolated new cohort, runs and configs preserve previous evidence. No branch/commit operations apply.
- Ruling: implement latest design using locally available sources; EyeQ and its ordinal loss deferred because no training originals exist. Cost: no EyeQ human-grade validation until a future release.
- Interface scan: cache requires frozen release; quality training requires complete bound cache; launch follows cache+GPU preflight gates. Existing preflight16 examples yields one batch at batch16, so increase to64 for genuine next-step resume checks.
- Task1 complete: CUDA tensor execution passed; frozen manifest byte-identical reviewed proposal. All17,697 unique source paths SHA256/full decode passed; all12 input fingerprints independently reproduced; zero known group/edge/content role crossings.
- Task2 complete: two initial regressions observed failing (missing partial-reuse parameter, workers still0), then33 tests passed. CPU cache seed rebound exactly2246 identities; metadata remains complete=false,expected_rows13789.
- Ruling: disease training can precede new quality target generation because A/B do not consume teacher targets. Pipeline order A→B→teacher cache→quality GPU preflight→quality. Cost: expanded quality readiness remains pending until its guarded stage; no claim that all three tasks are already ready.
- Review found two Important issues: supervisor failure could orphan writer processes; original real-GPU preflight never crossed an epoch. Fix pass: Windows kill-on-close Job Object with real child/abrupt-parent termination tests, and compare continuous/resumed first optimizer step of epoch1 using2epoch preflight. Missing cross-epoch gate test observed failing before fix. Original same-epoch A/B proof passed with0.0 parameter/loss differences; preserved as historical evidence. New cross-epoch proof running in a separate destination.
- Review fix pass complete: delayed-job-assignment race reproduced then fixed with CREATE_SUSPENDED→assign→ResumeThread;37 tests pass. A/B cross-epoch proof passed with0.0 weight/loss differences, step5epoch1;2epochs8steps completed. Independent reviewer regraded both findings resolved and disease launch PASS; quality explicitly pending.
- Task3 ongoing by guarded pipeline:2246 targets rebound,11543 new targets pending. No ordinal supervision without EyeQ originals.
- Task4 disease complete; quality deferred to its own guarded pipeline stage. Review gate binds22 code/config/data/proof inputs.
- Task5 started06:21UTC with durable supervisorPID43020. Initial sandbox-contained background process did not survive command return and produced no training output; persistent external execution corrected process lifetime. Awaiting first full training epoch/validation stability observation.
- Task5 startup/stability verified06:24:39UTC: Acompleted1epoch and advanced tostep976epoch1. Full1190-image validation identity set correct; finite logs/gradients/predictions, contiguous steps, best/last present, checkpoint fingerprint correct, all reviewed inputs unchanged. Snapshot runs/formal_v3/pipeline/launch_health.json. Full20epochA/B/quality completion remains ongoing in the durable queue; tasks3/qualitypart4 remain pending until their stages execute.
