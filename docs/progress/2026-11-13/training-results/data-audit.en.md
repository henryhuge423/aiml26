# Expanded training cohort: data integration audit

November 13, 2026 · Eye Sentinel

[中文](data-audit.zh.md)

This integration jointly groups the 9,070 admitted November 13 photographs with the original formal_v3 disease cohort and creates a separate frozen release, `expanded_20261113`. One manifest assigns explicit training, model-validation, rule-development, and test roles. Original data retain their paths and historical reports are archived by date; the new release records exclusions, input hashes, and verification results separately.

## Findings and corrections before training

The audit rebuilds the transitive closure of photograph and patient relationships: photographs connected directly or through other images share one group. The graph includes original formal_v3 groups, confirmed November 13 associations and exact hashes, known patients, historical confirmed edges, EyeQ patient groups, and original MSHF TIFF files linked to their equal-pixel PNG copies.

This check found 18 original disease photographs associated with official EyeQ test photographs: 16 training images and two model-validation images. They are excluded from the new disease release so that these EyeQ groups retain their test role in the new release. Another 12 original rule-development images are removed according to the November 13 identity-based repair. The removed training images comprise 13 positives and three negatives; both removed validation images are positive. The retained original training portion contains 10,338 images and 4,409 positives; original model validation contains 1,188 images and 487 positives.

The original 1,233 internal-test, 705 ACRIMA external-test, and 737 HYGD external-test images retain their identities, labels, and roles: 2,675 protected disease images in total. Verification reads images and metadata without model predictions on the test sets.

## Allocation of new photographs

All 9,070 additions retain original labels, source-defined disease endpoints, imaging domains, author partitions, and official quality annotations. The 204 suspects remain separate from binary disease training. KEH photographs retain their optic-disc-crop domain. MSHF files containing TIFF data use verified equal-RGB PNG training copies, with the originals preserved.

| Role | Original images | Added images | Total images | Total positives |
|---|---:|---:|---:|---:|
| Training | 10,338 | 5,142 | 15,480 | 6,405 |
| Model validation | 1,188 | 858 | 2,046 | 823 |
| Rule development | 1,220 | 833 | 2,053 | 814 |
| Internal and added held-out tests | 1,233 | 2,237 | 3,470 | 1,292 |
| ACRIMA external test | 705 | 0 | 705 | 396 |
| HYGD external test | 737 | 0 | 737 | 540 |

The additions form 8,434 linked groups. Whole groups are stratified by source, label, and real poor-quality positive status, using seed 20261113. LAG, GRAPE, FIVES, Fundus-AVSeg, and MSHF relationships are processed within one graph, so linked photographs share one role across sources. Where patient identifiers are unavailable, evaluation uses confirmed image-association groups; these are image-group counts, while patient counts are reported for sources with patient identifiers.

All 420 PAPILA photographs are held out as a whole-source test, including 87 positive images from 47 known patient groups. The 97 official FIVES test images and eight official Fundus-AVSeg test images remain in test, together with their linked patient or photograph groups. Test photographs are excluded from training-parameter and checkpoint selection.

## Real poor-quality positives

| Source and official definition | Training | Model validation | Rule development | Test |
|---|---:|---:|---:|---:|
| FIVES: at least one poor quality dimension | 44 | 7 | 7 | 41 |
| MSHF: poor official overall quality | 28 | 5 | 5 | 9 |
| Total | 72 | 12 | 12 | 50 |

FIVES stores the condition as `True` in `quality_fives_any_poor_dimension`; MSHF stores poor overall quality as `0` in `quality_overall_official`. Their definitions differ, so evaluation reports each source separately. Model validation and rule development each contain only 12 such positives: one decision changes sensitivity by about 8.3 percentage points. Training comparisons therefore retain detected-image counts alongside sensitivity.

## Verification and training interface

All 24,491 disease images pass file SHA256, full RGB decoding, and data-directory boundary checks. All 9,070 additions also match the original pixel hashes recorded in the November 13 audit. MSHF training copies are rechecked against the decoded original TIFF pixels. There are zero group crossings, zero cross-role equal-RGB hashes, and zero duplicate file hashes. Three unit checks pass, covering transitive grouping, whole-group real-quality stratification, and source-specific quality codes.

The unified manifest adds `origin_cohort`, with values `formal_v3` and `expansion_20261113`. The original-data control and expanded-data experiments can select training rows using this field while sharing model-validation and rule-development rows. Group corrections apply to the original-data control and expanded arms, which share the corrected validation and development roles. The existing quality model remains frozen and uses original-image quality supervision; synthetic degradation supports disease training.

Reproduction commands:

```powershell
tools/training_python.cmd tools/freeze_expansion_training_20261113.py --verify
tools/training_python.cmd -m unittest discover -s tests -p test_expansion_cohort_20261113.py
```

[Frozen manifest](../../../../data/prepared/expanded_20261113/manifest.csv) · [Release and input binding](../../../../data/prepared/expanded_20261113/release.json) · [Exclusions](../../../../data/prepared/expanded_20261113/exclusions.csv) · [Verification summary](data-cohort-summary.json) · [Freezer](../../../../tools/freeze_expansion_training_20261113.py)
The [training results](training-results.en.md) follow this audit. The [data and degradation review](../data-review/data-and-degradation-review.en.md) explains source selection.
