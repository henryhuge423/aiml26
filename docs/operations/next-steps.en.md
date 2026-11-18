# Next experiments and evaluation

Updated November 16, 2026 · [中文](next-steps.md)

All 18 training and development evaluations are complete: six conditions with three common seeds, each trained for 2,400 parameter updates. The [full analysis](../progress/2026-11-15/completed-experiments/completed-results.en.md) provides ten figures comparing selected and end-of-budget checkpoints, source sampling, actual augmentation exposure, classification thresholds, and quality-guided review. The release is `expanded_20261113`; see [completion status](../../runs/followup_20261115/pipeline/status.json).

Under the existing five-source validation criterion, 65% original-source sampling plus mixed degradation is the candidate for further work, with three-seed mean 0.9634. Seed 20261117 at update 2,100 has the highest validation value within this condition, 0.9665. The fixed-boundary rehearsal used this checkpoint, the historical ResNet-50, and a same-classifier uncertainty control at about 10% review. The pooled and AIROGS-only results had opposite directions. Specify the intended screening task and primary cohort before final testing. [Proposal review and fixed-boundary rehearsal](../progress/2026-11-16/proposal-review.en.md)

| Priority | Work | Evidence and evaluation |
|---|---|---|
| Acquire real poor-quality cases | Prioritize independent negatives from the same source and collection conditions, with explicit negative glaucoma diagnoses and human quality grades; expand the poor-quality positive evaluation stratum as well. | Poor-quality training contains 72 positives and three negatives; rule development contains 12 positives and no negatives. Estimate misses and false alarms separately. |
| Fix disease thresholds | Define the calibration cohort for the intended use, then fix thresholds on its model-validation data. | The rehearsal used 95% validation specificity with pooled and AIROGS-only calibration. Report denominators, including ODIR's 28 development positives across 19 positive linked groups. |
| Fix the review rule | Use about 10% as the development calibration target and fix one review cutoff. Match both uncertainty baselines to the boundary's actual workload on test. | At 206 reviews on development, joint quality left three fewer errors than same-classifier uncertainty, with a paired interval spanning zero. |
| Evaluate real degradation | Choose degradation type and severity using same-eye retakes and optic-disc visibility. | Three matched-defocus runs are complete. The gains appeared mainly on synthetic defocus. [Defocus comparison](../progress/2026-11-16/quality-alignment.en.md) |
| Review diagnostic visibility | Record overall photographic quality, visibility of diagnostic optic-disc information, and error type during human review. | FTHNet and EyeQ distinguish human quality grades; their additional ranking information for disease errors is weak. |

Plan real poor-quality acquisition in independent cases. Under zero errors, at least 59 positives and separately 59 negatives would place each one-sided 95% error-rate bound below 5%; zero misses in the current 12 positives gives a 22.1% upper bound. Only 5/150 original normal photographs in the FIVES official training partition have poor quality. Local MSHF Healthy photographs all have good overall quality and were excluded for duplicates or conflicting labels during the audit. Source and quality composition remain part of acquisition planning.

## Complete the main project comparison

1. Select disease model A, comparator B, quality model, and classification thresholds using model validation; record preprocessing and checkpoints.
2. Select review regions in rule development, reporting case counts, remaining errors, and misses by source, disease class, and human quality grade. Use group cross-validation for fitted error ranking.
3. Specify the workload and miss-rate evaluation criteria; fix weights, classification thresholds, review regions, B's uncertainty ranking, and tie rules.
4. Evaluate the frozen reserved test allocations. If A refers k images, B selects the same k by uncertainty. Calculate paired intervals by patient or linked group.
5. Prepare final Chinese and English Markdown, PDF, LaTeX, and figures, reviewing every page for language, data, denominators, and layout.

Methods are documented in the [training and evaluation protocol](training-protocol-2026-11-09.en.md); earlier sampling and degradation results appear in the [November 13 statistical analysis](../progress/2026-11-13/experimental-analysis/experimental-analysis.en.md).
