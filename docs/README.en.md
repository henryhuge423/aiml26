# Documentation guide

English | [中文](README.md)

Start with the [latest progress overview](progress/2026-11-13/progress-overview.en.md), then follow the data review, training results, and statistical analysis as needed. Operational documents describe current entry points; historical records retain the versions and context of earlier stages.

| Category | Entry point | Contents |
|---|---|---|
| Progress reports | [Index by date](progress/README.en.md) | Bilingual Markdown, PDF, LaTeX, figures, and summary data. |
| Current operations | [Runbook](operations/training-runbook.en.md) | Current release, configurations, completion checks, and report generation. |
| Planned work | [Next steps](operations/next-steps.en.md) | Sampling and degradation controls, model selection, review rules, and final evaluation. |
| Readiness checks | [Training readiness](operations/training-readiness.en.md) | Evidence for current experiment inputs and completion. |
| Methods protocol | [Training and evaluation protocol](operations/training-protocol-2026-11-09.en.md) | Scores, data roles, manual-review comparison, and evaluation measures. |
| Data research | [Expansion research](research/data-expansion-2026-11-11.md) | November 11 source research, with the [outline](research/data-expansion/outline.yaml) and [field definitions](research/data-expansion/fields.yaml). |
| Historical records | [Historical index](historical/README.en.md) | Earlier proposals, audits, training snapshots, and original operational documents. |
| Repository layout | [Directory guide](DIRECTORY.en.md) | Locations of code, data, run artifacts, and project background. |

Folders under `progress/` use `YYYY-MM-DD`. Related work on one date shares a directory, with lowercase topic names separated by hyphens. Main reports use `.zh` and `.en` language suffixes and retain matching `.md`, `.pdf`, and `.tex` files. November 13's `data-review/`, `training-results/`, and `experimental-analysis/` cover data and methods, experiment execution, and interpretation.

Historical material retains its original findings and wording, with locations and references updated. `notes/` holds course reviews and early research notes; `archive/` holds earlier proposals. Dataset, training-configuration, and run paths are preserved.
