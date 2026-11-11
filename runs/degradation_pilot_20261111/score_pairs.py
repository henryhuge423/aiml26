"""Score paired synthetic variants of the repaired development images."""

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from tools.score_repaired_boundary import InferenceRows, predict, read
from training.data import ImageRows, sha256
from training.engine import fingerprint_inputs
from training.models import create_model


SEED = 20261111
PER_SOURCE_LABEL = 32
DISPLAY_THRESHOLD = 0.5
# Blur radius is specified relative to the longer edge of a 512 px image.
VARIANTS = {
    "original": ("original", 0.0),
    "blur_mild": ("blur", 2.0),
    "blur_strong": ("blur", 5.0),
    "dark_mild": ("dark", 0.70),
    "dark_strong": ("dark", 0.40),
}


def sample_originals(rows):
    ordered = sorted(
        rows, key=lambda row: hashlib.sha256(f"{SEED}:{row['sample_id']}".encode()).digest()
    )
    count = Counter()
    groups = set()
    selected = []
    for row in ordered:
        stratum = row["source"], row["label"]
        group = row["related_group"]
        if count[stratum] < PER_SOURCE_LABEL and group not in groups:
            selected.append(row)
            count[stratum] += 1
            groups.add(group)
    return selected


def image_variants(image):
    for name, (effect, strength) in VARIANTS.items():
        if effect == "blur":
            output = image.filter(ImageFilter.GaussianBlur(strength * max(image.size) / 512))
        elif effect == "dark":
            output = ImageEnhance.Brightness(image).enhance(strength)
        else:
            output = image
        yield name, output


class PairedImages(torch.utils.data.Dataset):
    def __init__(self, rows, config, save_example):
        self.rows = rows
        self.task = config["task"]
        self.base = ImageRows(rows, ROOT / "data", config["image_size"], self.task, train=False)
        self.save_example = save_example

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.base.root / row["path"]).resolve()
        assert path.is_relative_to(self.base.root) and sha256(path) == row["sha256"]
        with Image.open(path) as original:
            image = original.convert("RGB")
        tensors = []
        for name, variant in image_variants(image):
            if self.save_example and index == 0:
                thumbnail = ImageOps.contain(variant, (480, 480))
                thumbnail.save(OUT / f"example_{name}.png")
            if self.task == "glaucoma":
                side = max(variant.size)
                variant = ImageOps.pad(variant, (side, side), color=0)
            tensors.append(self.base.transform(variant))
        return torch.stack(tensors)


def score(rows, all_rows, model_name, key, verification):
    config = json.loads((ROOT / f"configs/formal_v3/{model_name}.json").read_text())
    checkpoint = ROOT / config["run_dir"] / "best.pt"
    evidence = next(item for item in verification["models"] if item["model"] == model_name)
    assert sha256(checkpoint) == evidence["checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["fingerprint"] == fingerprint_inputs(config, ROOT) == evidence["training_fingerprint"]
    model = create_model(config["architecture"], ROOT / "models/checkpoints").to("cuda")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    dataset = PairedImages(rows, config, save_example=key == "p_a")
    batches = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0)
    chunks = []
    with torch.inference_mode():
        for batch in batches:
            predictions = []
            for column in range(len(VARIANTS)):
                values = model(batch[:, column].contiguous().to("cuda")).flatten()
                if key == "p_a":
                    values = values.sigmoid()
                predictions.append(values.cpu().numpy())
            chunks.append(np.column_stack(predictions))
    scores = np.concatenate(chunks)
    assert np.isfinite(scores).all()
    reference = np.array([float(row[key]) for row in rows])
    differences = np.abs(scores[:, 0] - reference)
    error = float(differences.max())
    indices = {row["sample_id"]: i for i, row in enumerate(all_rows)}
    controls = np.argsort(differences)[-4:]
    original_adapter = InferenceRows(rows, ROOT / "data", config["image_size"], config["task"])
    original_batch_errors = []
    for i in controls:
        row = rows[i]
        assert torch.equal(dataset[i][0], original_adapter[i][0])
        sample_id = row["sample_id"]
        start = indices[sample_id] // 16 * 16
        original_scores = predict(model, all_rows[start:start + 16], config)
        original_batch_errors.append(abs(original_scores[sample_id] - reference[i]))
    old_batch_error = max(original_batch_errors)
    print(f"{model_name}: new batch max diff {error:.9g}, original batch max diff "
          f"{old_batch_error:.9g}", flush=True)
    assert old_batch_error < 1e-6, (model_name, old_batch_error)
    assert np.array_equal(scores[:, 0] >= DISPLAY_THRESHOLD, reference >= DISPLAY_THRESHOLD) or key == "q"
    return scores, {
        "original_score_max_abs_diff": error,
        "original_batch_control_max_abs_diff": old_batch_error,
        "original_batch_controls": len(controls),
        "checkpoint_sha256": sha256(checkpoint),
    }


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    boundary = ROOT / "runs/formal_v3/boundary_development_repaired_20261110"
    verification = json.loads((boundary / "verification.json").read_text())
    score_path = boundary / "boundary_pq.csv"
    assert sha256(score_path) == verification["boundary_scores_sha256"]
    all_rows = read(score_path)
    assert len(all_rows) == verification["rows"] == 1230
    assert all(row["task"] == "glaucoma" and row["split"] == "boundary" for row in all_rows)
    rows = sample_originals(all_rows)
    print(f"Selected {len(rows)} development originals", flush=True)
    p, p_proof = score(rows, all_rows, "glaucoma_a", "p_a", verification)
    q, q_proof = score(rows, all_rows, "quality_student", "q", verification)
    base = p[:, 0] >= DISPLAY_THRESHOLD
    positive = np.array([row["label"] == "1" for row in rows])
    results = {}
    records = []
    for j, name in enumerate(VARIANTS):
        delta_q = q[:, j] - q[:, 0]
        predicted = p[:, j] >= DISPLAY_THRESHOLD
        results[name] = {
            "median_delta_q": float(np.median(delta_q)),
            "q_decreases": int((delta_q < 0).sum()),
            "median_abs_delta_p": float(np.median(abs(p[:, j] - p[:, 0]))),
            "threshold_crossings": int((base != predicted).sum()),
            "positive_label_down": int((positive & base & ~predicted).sum()),
            "positive_label_up": int((positive & ~base & predicted).sum()),
            "negative_label_down": int((~positive & base & ~predicted).sum()),
            "negative_label_up": int((~positive & ~base & predicted).sum()),
            "positive_label_below_threshold": int((positive & ~predicted).sum()),
            "negative_label_at_or_above_threshold": int((~positive & predicted).sum()),
        }
        for i, row in enumerate(rows):
            records.append({
                **{key: row[key] for key in (
                    "sample_id", "sha256", "source", "label", "target_definition", "related_group"
                )},
                "variant": name,
                "p_a": float(p[i, j]),
                "q": float(q[i, j]),
                "delta_p_a": float(p[i, j]) - float(p[i, 0]),
                "delta_q": float(delta_q[i]),
            })
    assert len(records) == len(rows) * len(VARIANTS)
    assert all(math.isfinite(row["p_a"]) and math.isfinite(row["q"]) for row in records)
    with (OUT / "paired_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "originals": len(rows),
        "variants_excluding_original": len(rows) * (len(VARIANTS) - 1),
        "positive_labels": int(positive.sum()),
        "negative_labels": int((~positive).sum()),
        "counts_by_source_label": {
            f"{source} / {label}": count
            for (source, label), count in sorted(Counter((r["source"], r["label"]) for r in rows).items())
        },
        "sample_seed": SEED,
        "example_sample_id": rows[0]["sample_id"],
        "display_threshold": DISPLAY_THRESHOLD,
        "variants": VARIANTS,
        "boundary_scores_sha256": sha256(score_path),
        "paired_scores_sha256": sha256(OUT / "paired_scores.csv"),
        "p_a": p_proof,
        "q": q_proof,
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
