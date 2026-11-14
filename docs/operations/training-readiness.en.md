# Training readiness and completion

Updated November 13, 2026 · [中文](training-readiness.md)

`expanded_20261113` has completed data verification, GPU recovery preflight, four training arms, and development evaluation. Readiness checks establish that data and code support the specified experiment; result reports describe performance on identified evaluation sets.

| Check | Current evidence |
|---|---|
| Data release | 24,491 disease images; manifest hash recorded in [cohort_verification.json](../../data/prepared/expanded_20261113/cohort_verification.json). |
| Allocation | 15,480 training, 2,046 model validation, 2,053 rule development, 3,470 internal/added reserved tests, 705 ACRIMA, and 737 HYGD images. |
| Identity groups | Known patients and confirmed linked images share a role; file-hash and RGB checks cover every image. |
| Degradation | Illumination, defocus, scattering, and real-pair calibration documented in the [degradation audit](../progress/2026-11-13/training-results/degradation-audit.en.md). |
| GPU recovery | Continuous and interrupted runs of six updates have maximum weight difference 0 and identical loss sequences. |
| Completion | Four arms each reached 1,200 updates; see [execution status](../../runs/expanded_20261113/pipeline/status.json). |
| Evaluation | Common original development images and fixed synthetic stress subset, with separate selected and budget-end results. |

The [runbook](training-runbook.en.md) provides commands, and the [training report](../progress/2026-11-13/training-results/training-results.en.md) explains results. New configurations and releases require checks matching their own inputs. The original `formal_v2` readiness document is retained in [history](../historical/2026-11-09/training-readiness.md).
