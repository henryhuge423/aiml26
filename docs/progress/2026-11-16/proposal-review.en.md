# Eye Sentinel Proposal Review and Fixed-Boundary Rehearsal

November 16, 2026 · [中文](proposal-review.zh.md) · [PDF](proposal-review.en.pdf)

## Progress against the proposal

| Proposal task | Evidence and current status |
|-----------------------|---------------------------------------------------------------------------|
| Disease score $p$ | EfficientNet-B0 experiments compared data expansion, source sampling, and augmentation. Expansion improved ranking on LAG and KEH across three seeds. After matching defocus exposure in training, the clearest gain remained on synthetic defocus. A separately trained ResNet-50 is available as a disease baseline. |
| Quality score $q$ | The project trained a student distilled from FTHNet and a joint quality model that also uses EyeQ's three human grades. Both provide whole-image quality scores. Rule development includes 12 human-labeled poor-quality positives and zero poor-quality negatives under the available definitions. |
| Fixed boundary and matched review volume | This round applied a fixed cutoff to held-out development folds and matched the resulting review volume for the same-classifier and independent-classifier baselines. Page 4 of the proposal shows approximately 10% review. Pages 1–2 specify matching the actual volume selected by the boundary. |
| Independent testing and retakes | There are 3,470 internal and new held-out images, 705 ACRIMA images, and 737 HYGD images. This round scored only validation and rule-development images. The retake branch requires first-capture and repeat photographs of the same eye to estimate how often a repeat restores gradability. |

[Project proposal](../../../proposal-1106.pdf) · [Three-seed results](../2026-11-15/completed-experiments/completed-results.en.md) · [Defocus exposure and quality analysis](quality-alignment.en.md)

## Comparing fixed operating points

Model A follows the November 15 selection practice. It is the EfficientNet-B0 trained with 65% original-source sampling and 50% mixed degradation, seed 20261117, at update 2,100. Model B is the existing ResNet-50 trained for 20 epochs on the original cohort. Epoch 18 was selected by AUROC on the original AIROGS validation subset. This round scored B on 2,046 model-validation images and 2,053 original rule-development images. The main quality score is the continuous output of the joint EyeQ and FTHNet model, consistent with the quality training shown in the proposal diagram. A separate FTHNet-distilled student provides a sensitivity analysis.

Each classifier's disease threshold $t$ is the 95th percentile of pooled validation negatives. A score strictly above $t$ predicts a positive. This was an analysis choice. AIROGS also reports sensitivity at 95% specificity, and our percentile combines negatives across sources. The development set contains 2,053 original images, including 814 positives, across 1,917 linked groups. Five folds preserve groups and approximately balance sources and labels. The review score is $-|\operatorname{logit}(p)-\operatorname{logit}(t)|+(1-q)$. Images near the disease threshold receive higher priority. Low quality widens that region by up to one log-odds unit. We clip raw quality scores to the interval from 0 to 1 for this calculation. Eight joint-model development scores fell outside that interval. All distilled scores were within it.

For each fold, the other four folds set the review cutoff at a workload of about 10%. The cutoff then applies directly to the held-out fold. After pooling the five folds, each baseline reviews the same number $k$, ranking images by how close each disease score is to that classifier's threshold. Sample identifier breaks ties. The automatic error rate uses unreviewed images as the denominator, and the residual miss rate uses all 814 positive images. The 95% intervals use 2,000 paired resamples of linked groups. Each resample rematches the baseline to the boundary's realized review volume, conditional on the current models and computed out-of-fold decisions.

[AIROGS evaluation methods](https://arxiv.org/html/2302.01738) · [Analysis script](../../../tools/rehearse_proposal_review_20261116.py)

## Development results

The disease thresholds were 0.7528 and 0.5813. On model validation, both reached 95.0% specificity, with sensitivities of 83.2% and 75.6%, respectively. The joint quality boundary selected 206 development images (10.03%) for review and left 1,847 automatic decisions.

| Review method | Reviewed | Remaining errors (rate) | Misses | False positives |
|-------------------------------|----------:|----------------------:|-----------:|--------------:|
| A with joint quality | 206 | 102 (5.52%) | 71 | 31 |
| A by disease uncertainty | 206 | 105 (5.68%) | 74 | 31 |
| B by disease uncertainty | 206 | 159 (8.61%) | 134 | 25 |

**The pooled difference mostly came from the disease classifier.** Compared with B, A with joint quality left 57 fewer errors and 63 fewer misses, with six more false positives. The paired difference in automatic error rate was −3.09 percentage points, with a 95% interval of [−4.61, −1.60]. A using disease uncertainty alone already left 54 fewer errors than B. Within A at matched volume, joint quality removed three additional errors, all misses. The error rate difference was −0.16 percentage points, with an interval of [−0.44, +0.17]. The distilled score left one additional error compared with A alone at 207 reviews. At a disease threshold of 0.5, joint quality left two additional errors and one additional miss compared with A alone. The incremental value of whole-image quality varied with the quality model and the disease threshold.

**The direction reversed within AIROGS.** In the mixed cohort, A with joint quality left 59 fewer errors than B across LAG and KEH combined, and 26 more on AIROGS. B's original training did not include LAG or KEH. After observing this difference, we recalibrated disease thresholds on 767 AIROGS validation images and repeated grouped holdout on the 773 AIROGS development images. At 77 reviews each, A with quality, A alone, and B left 46, 45, and 23 errors, with 31, 28, and 14 misses, respectively. A with quality left 23 more errors than B, with a 95% group-resampling interval of [11, 37]. This exploratory result shows how much the pooled advantage depends on the source mix.

All 12 human-labeled poor-quality positives stayed in the automatic branch and were correctly classified by A at this threshold. This small subgroup cannot estimate false positives on poor-quality negatives or outcomes after a retake. The review table estimates errors remaining in automatic decisions. Reader accuracy during actual review remains to be measured.

## Moving to independent evaluation

1. **Lock the screening comparison.** Set the primary cohort according to the intended use. For AIROGS-style screening, use a cohort with a referral endpoint. For cross-source classification, report source and label definitions and retain ACRIMA and HYGD as separate external cohorts. Before seeing test predictions, record A, B, the quality model, disease thresholds, boundary parameters, preprocessing, and tie handling. Estimate one final review cutoff on the designated development cohort. On the independent test, A's fixed boundary determines $k$, and both uncertainty baselines review $k$ images. The same-A comparison directly estimates the contribution of quality. A ResNet-50 trained on the same expanded data would help disentangle the training-data difference in the cross-source comparison.
2. **Label quality for glaucoma grading.** Prioritize real first-capture failures and repeat photographs of the same eye with an independent disease reference, especially poor-quality negatives. Blinded graders should record visibility of the disc rim and cup as well as the location of blur, underexposure, or obstruction. The existing AIROGS NRG candidates can support a grading exercise and workload estimate. NRG denotes a non-referral label. The REFUGE2 onsite set reports 76 Reject images, some with blur or artifacts outside the disc. Confirm access to per-image grades, independence, and usage terms before adding it. Retain the reference used to establish each label. [Existing candidate inventory](quality-alignment.en.md#implications-for-the-next-experiments) · [REFUGE2 paper, section 5.2](https://arxiv.org/html/2202.08994#S5.SS2)
3. **Measure retakes separately from routing.** The current data measure how many errors are removed from automatic decisions and how many misses and false positives remain. A retake study needs the reason for the first failure, the number of attempts, the fraction that regain gradability, final referral disposition, and time. Real Fundus pairs can check quality ranking and disc visibility. Estimating misses after a repeat requires pairs with an independent glaucoma reference. [Real Fundus paper](https://arxiv.org/abs/2201.00466)

[Results, source breakdowns, and input hashes](proposal-review-results.json)
