# Training readiness and completion

Updated November 15, 2026 · [中文](training-readiness.md)

`expanded_20261113` has completed data verification, GPU recovery preflight, four training arms, and development evaluation.

`followup_20261115` uses the same frozen cohort. All 18 training and development evaluations are complete: six conditions with three seeds, each trained for 2,400 parameter updates. CUDA uninterrupted/resume checks passed for mixed degradation, 65% original-source sampling, and defocus-only augmentation, with zero maximum weight difference and identical step records and state. See [completion status](../../runs/followup_20261115/pipeline/status.json), [preflight evidence](../../runs/followup_20261115/preflight/verification.json), and the [full results analysis](../progress/2026-11-15/completed-experiments/completed-results.en.md).

| Check | Current evidence |
|---|---|
| Data release | 24,491 disease images; manifest hash recorded in [cohort_verification.json](../../data/prepared/expanded_20261113/cohort_verification.json). |
| Allocation | 15,480 training, 2,046 model validation, 2,053 rule development, 3,470 internal/added reserved tests, 705 ACRIMA, and 737 HYGD images. |
| Identity groups | Known patients and confirmed linked images share a role; file-hash and RGB checks cover every image. |
| Degradation | Illumination, defocus, scattering, and real-pair calibration documented in the [degradation audit](../progress/2026-11-13/training-results/degradation-audit.en.md). |
| GPU recovery | Continuous and interrupted runs of six updates have maximum weight difference 0 and identical loss sequences. |
| Completion | Four November 13 conditions reached 1,200 updates; all 18 November 15 runs reached 2,400 updates and completed development evaluation. |
| Evaluation | Common original development images and fixed synthetic stress subset, with separate selected and budget-end results. |

The [runbook](training-runbook.en.md) provides commands, and the [training report](../progress/2026-11-13/training-results/training-results.en.md) explains results. New configurations and releases require checks matching their own inputs. The original `formal_v2` readiness document is retained in [history](../historical/2026-11-09/training-readiness.md).
