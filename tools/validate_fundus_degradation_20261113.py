"""Compare bounded augmentation with real repeated photographs on CPU.

Image statistics describe acquisition changes; they are not glaucoma labels or
clinical assessability grades. Both paired archives were already development
material in the November 11 pilot and are not final-test cohorts.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.fundus_degradation import FAMILIES, FundusDegradation, retinal_context

OUT = ROOT / "docs/progress/2026-11-13/training-results"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 ** 2):
            h.update(chunk)
    return h.hexdigest()


def pairs(archive):
    files = {Path(name) for name in archive.namelist()
             if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"} and "__MACOSX" not in Path(name).parts}
    for high in sorted(p for p in files if p.parent.name == "gt" or p.stem.endswith("_good")):
        if high.parent.name == "gt":
            low = high.parent.parent / "input" / high.name
            identity = high.stem
        else:
            identity = high.stem.removesuffix("_good")
            matches = [p for p in files if p.stem == identity + "_bad"]
            if len(matches) != 1:
                raise ValueError("same-eye pair missing or ambiguous")
            low = matches[0]
        if low not in files:
            raise ValueError("low-quality partner missing")
        yield identity, high.as_posix(), low.as_posix()


def read(archive, name):
    with archive.open(name) as stream, Image.open(stream) as image:
        image.load()
        image = image.convert("RGB")
    side = max(image.size)
    return ImageOps.pad(image, (side, side), color=0).resize((320, 320), Image.Resampling.BILINEAR)


def image_statistics(image):
    rgb = np.asarray(image).astype(np.float32) / 255
    mask = cv2.erode((rgb.max(2) > 8 / 255).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    green = rgb[..., 1]
    if not mask.any():
        raise ValueError("empty fundus field")
    luminance = .2126 * rgb[..., 0] + .7152 * green + .0722 * rgb[..., 2]
    return dict(mean_luminance=float(luminance[mask].mean()),
                green_contrast_std=float(green[mask].std()),
                green_laplacian_variance=float(cv2.Laplacian(green, cv2.CV_32F)[mask].var()),
                field_diameter_pixels=retinal_context(image)["field_diameter_pixels"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    augmentation = FundusDegradation(1)
    family_seeds = {}
    probe = Image.new("RGB", (320, 320), (150, 75, 45))
    for seed in range(100):
        _, record = augmentation.augment(probe, seed=seed)
        family_seeds.setdefault(record["family"], seed)
        if len(family_seeds) == len(FAMILIES):
            break
    datasets = {"HRF same-eye development": (ROOT / "data/external_quality/HRF_allQuality.zip", 18),
                "Real Fundus development": (ROOT / "data/external/real_fundus/Real_Fundus.zip", 120)}
    all_rows, summaries, previews, replay_examples = [], {}, [], []
    for source, (path, expected) in datasets.items():
        with zipfile.ZipFile(path) as archive:
            subjects = list(pairs(archive))
            if len(subjects) != expected:
                raise ValueError("pair count differs from prior audit")
            for index, (identity, high_name, low_name) in enumerate(subjects):
                high, low = read(archive, high_name), read(archive, low_name)
                versions = {"high": high, "real_low": low}
                for family in FAMILIES:
                    changed, record = augmentation.augment(high, seed=family_seeds[family])
                    versions[family] = changed
                    if index == 0:
                        replay_examples.append(dict(source=source, pair_id=identity, high_name=high_name, metadata=record))
                for version, image in versions.items():
                    all_rows.append(dict(source=source, pair_id=identity, version=version, **image_statistics(image)))
                if index == 0:
                    previews.append((source, identity, versions))
        rows = [r for r in all_rows if r["source"] == source]
        lookup = {(r["pair_id"], r["version"]): r for r in rows}
        output = dict(pairs=expected, archive_sha256=digest(path), comparison={})
        for version in ("real_low", *FAMILIES):
            comparison = {}
            for metric in ("mean_luminance", "green_contrast_std", "green_laplacian_variance"):
                ratio = np.array([lookup[(identity, version)][metric] / max(lookup[(identity, "high")][metric], 1e-10) for identity, _, _ in subjects])
                comparison[metric + "_ratio_to_high"] = dict(median=float(np.median(ratio)), p10=float(np.percentile(ratio, 10)), p90=float(np.percentile(ratio, 90)))
            output["comparison"][version] = comparison
        summaries[source] = output
        print(source, "completed", expected, flush=True)
    with (OUT / "degradation-pair-statistics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = dict(pillow_version=pillow_version, preprocessing="RGB -> square black pad -> bilinear 320x320 resize -> degradation",
                  statistic_role="Descriptive development image statistics; not clinical assessability or disease outcomes",
                  family_seeds=family_seeds, data=summaries, replay_examples=replay_examples)
    (OUT / "degradation-calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = ("high", "real_low", *FAMILIES)
    sheet = Image.new("RGB", (320 * len(columns), 370 * len(previews)), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for row, (source, identity, versions) in enumerate(previews):
        for column, name in enumerate(columns):
            x, y = 320 * column, 370 * row
            sheet.paste(versions[name], (x, y + 48))
            ImageDraw.Draw(sheet).text((x + 6, y + 5), f"{source.split(' development')[0]} {identity}\n{name}", font=font, fill="black")
    sheet.save(OUT / "degradation-preview.png")


if __name__ == "__main__":
    main()
