# Project directory guide

English | [中文](DIRECTORY.md)

This guide describes the Eye Sentinel repository. It sits in `team_project/` within the course workspace; course assignments and lectures outside it are not part of this repository.

| Location | Contents |
|---|---|
| `README.md`, `README.en.md` | Project overview and entry points. |
| `proposal-1106.pdf` | Current project proposal. |
| `docs/progress/` | Chinese and English progress reports, figures, and summary data; see the [report index](progress/README.en.md). |
| `docs/` | Training protocols, data research, experiment records, and historical material; `superpowers/` holds earlier plans and designs. |
| `configs/` | Data and training settings; `formal_v3/` contains the formal training configurations. |
| `training/` | Data loading, models, training, and validation code. |
| `tools/` | Data preparation, audits, training launchers, and other project tools. |
| `tests/` | Automated project checks. |
| `data/` | Dataset guides and audit records; local copies may also contain datasets, splits, and manifests. |
| `models/` | Model resources and weights; weights remain local. |
| `runs/` | Training status, metrics, checkpoints, and validation results; a small set of shareable summaries is tracked in Git. |
| `notes/` | Background, reviews, and early research notes. |
| `archive/` | Earlier proposals and historical materials. |
| `references/` | Local reference materials. |
| `tmp/`, `.cache/`, `.build/`, `.test_tmp/` | Temporary files, caches, and build outputs. |
| `.envs/` | Local runtime environments. |

GitHub holds the code, configurations, documents, and selected summaries. Raw data, model weights, complete run logs, local environments, and temporary files stay local under `.gitignore`. Existing directories retain their paths so that training configurations and document links continue to work. New progress reports belong in `docs/progress/`. Historical technical records retain their original language; new reports are provided in both languages.
