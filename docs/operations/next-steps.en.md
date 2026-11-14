# Next experiments and evaluation

Updated November 13, 2026 · [中文](next-steps.md)

Data integration, four training arms, and paired analysis are complete. Expansion improves added-source performance, while original-source presentations fall and AIROGS/ODIR ranking declines. Degradation benefits concentrate on synthetic defocus. The next round targets these observed differences.

| Work | Proposed comparison | Evaluation |
|---|---|---|
| Separate composition and sampling | Original-data control plus four expanded arms: current sampling / 65% original-data quota, each with no degradation / 50% mixed degradation. | Main-source AUROC, original-image probability error, actual sampling counts, and unique-image coverage. |
| Add repetitions and budget | Three common seeds, 2,400 updates each, weights and predictions saved every 300 updates. | Separate case-sampling intervals and seed variation; compare updates 1,200 and 2,400. |
| Improve checkpoint selection | Predeclare an equal AUROC mean over AIROGS, ODIR, Kaggle, LAG, and KEH; report other sources separately. | Also inspect the original three-source mean and source-threshold measures to reduce small-source influence. |
| Separate degradation mechanisms | Defocus-only versus mixed degradation at the same application probability, sampling sequence, and budget. | Misses and false positives at fixed validation thresholds, original AUROC, Brier score, and real poor-quality results. |
| Add real-quality cases | Same-source real poor-quality negatives, stratified by source, quality, and class. | Estimate false positives alongside detection and examine quality–disease label association. |

These are proposed experiments and have not yet run. Each round uses separate configurations, output directories, and data-version records, retaining this round as a control.

## Complete the main project comparison

1. Select disease models A and B and the quality model on model-validation data; record preprocessing, weights, classification thresholds, and calibration.
2. On rule-development data, examine joint disease and quality scores, showing correct predictions, errors, misses, and false positives with region counts.
3. Compare a limited set of review rules and select a boundary using workload, residual errors, and misses. Use grouped subsets or cross-fitting for calibration and repeated selection.
4. Lock weights, thresholds, the review boundary, B's uncertainty ranking, and tie rules before reserved-test evaluation. A's actual review count is k; B reviews the same k images by uncertainty.
5. Report review counts, residual automated error rates, and positive misses by source and quality stratum, with paired intervals based on patients or linked groups.

The workload–miss tradeoff still needs an explicit selection criterion. The [training and evaluation protocol](training-protocol-2026-11-09.en.md) describes the method, and the [statistical analysis](../progress/2026-11-13/experimental-analysis/experimental-analysis.en.md) provides current evidence. The [historical checklist](../historical/2026-11-10/next-steps.md) retains the November 10 plan.
