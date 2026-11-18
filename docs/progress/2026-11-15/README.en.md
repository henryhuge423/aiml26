# November 15 training and quality analysis

[中文](README.md)

## Results after training completion

[Completed three-seed analysis](completed-experiments/README.en.md) · [English Markdown](completed-experiments/completed-results.en.md) · [English PDF](completed-experiments/completed-results.en.pdf) · [LaTeX](completed-experiments/completed-results.en.tex)

All 18 training and development evaluations are complete. The new report adds ten bilingual figures covering sources, full validation curves, recorded augmentation exposure, classification thresholds, quality scores, and manual review at equal workloads.

## Quality analysis at training launch

[English report](quality-and-training.en.md) · [English PDF](quality-and-training.en.pdf) · [LaTeX](quality-and-training.en.tex)

[中文报告](quality-and-training.zh.md) · [中文 PDF](quality-and-training.zh.pdf) · [LaTeX](quality-and-training.zh.tex)

This work checks previous experiments, starts six training conditions with three seeds and 2,400 updates each, and analyzes the number of poor-quality cases, disease-label imbalance, source quality scales, and the incremental review value of p,q. Reported training progress uses the [saved snapshot](training-snapshot.json); later execution is recorded in [live status](../../../runs/followup_20261115/pipeline/status.json).

[Analysis data](quality-error-analysis.json) · [Run guide](../../../runs/followup_20261115/README.en.md) · [Previous experiments](../2026-11-13/README.en.md)
