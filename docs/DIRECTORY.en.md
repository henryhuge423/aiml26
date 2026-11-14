# Project directory guide

English | [中文](DIRECTORY.md)

The Eye Sentinel repository occupies `team_project/` within the course workspace. Documentation is organized into progress, current operations, research, and historical records. Dataset and training-program paths retain their locations.

| Location | Contents and entry points |
|---|---|
| `README.md`, `README.en.md` | Project goals, latest results, and main entry points. |
| `proposal-1106.pdf` | Current project proposal. |
| `docs/README.md`, `docs/README.en.md` | [Documentation guide](README.en.md). |
| `docs/progress/YYYY-MM-DD/` | Bilingual progress reports, PDFs, LaTeX, figures, and summaries by date. |
| `docs/progress/2026-11-13/` | [Combined dataset and degradation experiment materials](progress/2026-11-13/README.en.md), with `data-review/`, `training-results/`, and `experimental-analysis/`. |
| `docs/operations/` | Current runbook, readiness checks, protocol, and next steps in both languages. |
| `docs/research/` | Data-expansion research and structured outlines. |
| `docs/historical/` | [Dated audits and earlier operations](historical/README.en.md), earlier dataset introduction, and speaker notes. `planning/` retains original designs and execution plans. |
| `configs/` | Training configurations; `formal_v3/` is earlier training, and `expanded_20261113/` contains this round's four arms. |
| `training/` | Data loading, models, training, quality supervision, and fundus degradation. |
| `tools/`, `tests/` | Data checks, training and report tools, and automated checks. |
| `data/` | [Data guide](../data/README.en.md), source files, manifests, and audits. |
| `models/` | [Model resource guide](../models/README.en.md), local initialization, teacher, and trained weights. |
| `runs/` | Training status, metrics, predictions, and weights; selected summaries are tracked in Git. |
| `notes/` | Early research, feasibility reviews, and course-review notes. |
| `archive/`, `references/` | Earlier proposals and local reference materials. |
| `.envs/` | Local execution environment. |
| `tmp/`, `.cache/`, `.build/`, `.test_tmp/` | Temporary files, caches, PDF renders, and check artifacts. |

Topic directories use lowercase names with hyphens; progress folders use dates. Historical text is retained, with references updated when moved. Raw data, weights, complete logs, environments, and temporary artifacts stay local. GitHub shares code, configurations, documents, PDFs, figures, and selected summaries.
