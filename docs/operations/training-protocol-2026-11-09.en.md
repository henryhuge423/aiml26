# Research training and evaluation protocol

Initial design: November 9, 2026; revised documentation: November 13 · [中文](training-protocol-2026-11-09.md)

The project studies disease classification and manual-review selection for fundus photographs. Disease model A produces glaucoma score p, and a quality model produces image-quality score q. A review rule selects photographs using p and q; A's classification remains for images outside the review region. The main comparator is model B selecting the same number of images by disease-prediction uncertainty.

## Data roles and scores

Training data update model parameters; model-validation data select checkpoints and classification thresholds; rule-development data define the manual-review boundary; reserved tests support final evaluation. Known patients and confirmed linked images are allocated by group. Source label definitions and imaging domains, including disc crops, are recorded separately. The [data audit](../progress/2026-11-13/training-results/data-audit.en.md) describes the current release.

Disease scores support ranking and threshold classification; quality scores describe image quality. The quality student learns fixed teacher targets, and EyeQ quality training also uses manual three-grade labels. Teacher-fitting error and manual-label performance are evaluated separately. Current disease augmentation includes illumination, defocus, and scattering; quality targets remain aligned with their original images.

## Select models and review rules

Select A, B, the quality model, preprocessing, and classification thresholds on model-validation data, recording checkpoint provenance and metrics. Then display case counts, accuracy, misses, and false positives in the rule-development p–q plane, specifying each region's denominator.

Choose a review boundary from a limited, predeclared set using residual errors, positive misses, and manual workload. The selected boundary determines review proportion. Calibration and rule selection use separate grouped subsets or cross-fitting. Boundary forms and workload-selection criteria are fixed before final evaluation.

## Match review counts

For N test images, A's rule selects k for review, giving review proportion k/N. B ranks images using a predefined uncertainty measure and reviews the same k images. Ties follow fixed, label-independent identifiers or seed rules. This matches B's review count to A's workload.

Primary measures are the error rate among N−k remaining automated results and positive miss counts and rates. Also report review counts, positive denominators, sensitivity, specificity, and source. Resampling uses patients or confirmed linked groups, with the same groups for both schemes. When A and B differ, the main comparison evaluates the complete schemes; an additional A-uncertainty control can examine the quality rule's contribution.

Lock all models, thresholds, review rules, and ranking rules before final evaluation. Data preparation and disease-model development experiments are complete; the main review comparison remains to be run. See [next steps](next-steps.en.md). The [November 9 historical protocol](../historical/2026-11-09/training-protocol-2026-11-09.md) retains the original preparation design.
