# Audited additive data pool

English | [中文](README.md) · November 13, 2026

The pool contains 9,070 images, including 3,446 positives, plus 204 separately recorded suspects. This directory records sources, labels, image links, and file identities at completion of source review. The [joint release](../expanded_20261113/README.en.md) records their subsequent roles on the same day: 5,142 training, 858 model validation, 833 rule development, and 2,237 reserved tests.

| Source | Images | Positives |
|---|---:|---:|
| airogs_original_rg | 3 | 3 |
| beh | 333 | 83 |
| chaksu | 861 | 0 |
| crfo | 67 | 40 |
| fives | 384 | 191 |
| fundus_avseg | 42 | 14 |
| grape | 518 | 518 |
| hrf | 30 | 15 |
| keh | 1,529 | 747 |
| lag_part1 | 4,812 | 1,688 |
| les_av | 22 | 11 |
| mshf | 49 | 49 |
| papila | 420 | 87 |

`pool_manifest.csv` records source labels, file and pixel hashes, photo/patient groups, and author partitions; `release.json` binds review inputs. LAG, GRAPE, FIVES, Fundus-AVSeg, and MSHF are grouped jointly. KEH retains its optic-disc crop domain. MSHF PNG training copies have the same decoded RGB pixels as their original TIFF files. Suspect images retain their labels in `suspect_manifest.csv`.

[Source-review report](../../../docs/progress/2026-11-13/data-review/data-and-degradation-review.en.md) · [Training data audit](../../../docs/progress/2026-11-13/training-results/data-audit.en.md)
