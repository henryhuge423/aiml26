"""Measure paired changes after the actual disease-training degradation."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from openpyxl import load_workbook
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_expansion_20261113 import poor
from tools.run_expansion_experiments_20261113 import write_json
from tools.run_followup_experiments_20261115 import SOURCES
from training.data import ImageRows, load_rows, sha256
from training.eyeq_quality import QualityModel
from training.fundus_degradation import FAMILIES, FundusDegradation, retinal_context
from training.models import create_model

COHORT = ROOT / "data/prepared/expanded_20261113"
OUTPUT = ROOT / "runs/quality_alignment_20261116"
TRAIN_SIZE = 320


def read(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def tail(values, cut=.2):
    values = np.asarray(values, dtype=float)
    lo, median, hi = np.quantile(values, [.1, .5, .9])
    return dict(n=len(values), below=int((values < cut).sum()),
                at=int((values == cut).sum()), cut=cut,
                minimum=float(values.min()), p10=float(lo), median=float(median),
                p90=float(hi), maximum=float(values.max()))


def inventories():
    scores = read("runs/followup_20261115/error_ranking/development_join.csv")
    official = read("data/external/hygd-1.1.0/Labels.csv")
    candidates = read("data/external/airogs_candidate_5k/exploratory_scores.csv")
    workbook = load_workbook(ROOT / "data/external/fiqs/meta_data.xlsx",
                             read_only=True, data_only=True)
    iterator = workbook.active.values
    header = next(iterator)
    fiqs = [dict(zip(header, row)) for row in iterator if row[1]]
    workbook.close()
    results = {}
    for role in ("val", "boundary"):
        rows = [r for r in scores if r["split"] == role]
        results[role] = {
            "student_q": tail([r["q"] for r in rows]),
            "eyeq_q": tail([r["q_eyeq"] for r in rows]),
            "eyeq_grade": dict(Counter(r["predicted_eyeq_grade"] for r in rows)),
            "by_label": {label: tail([r["q"] for r in rows if r["label"] == label])
                         for label in ("0", "1")},
            "by_source": {source: tail([r["q"] for r in rows if r["source"] == source])
                          for source in sorted({r["source"] for r in rows})},
        }
    pairs = read("runs/degradation_pilot_20261111/real_pairs_quality.csv")
    legacy_pairs = read("runs/degradation_pilot_20261111/paired_scores.csv")
    student_val = read("data/audit/session_20261110/selected_validation/quality_student/validation_predictions.csv")
    fiqs_val = [r for r in student_val if r["human_mos_normalized"]]
    eyeq_val = read("runs/eyeq_quality_v1/quality_student/best_validation_predictions.csv")
    result = dict(
        cohort=results,
        human_poor_by_role_label={role: dict(Counter(
            r["label"] for r in load_rows(COHORT / "manifest.csv", "glaucoma", role,
                                          COHORT / "release.json") if poor(r)))
            for role in ("train", "val", "boundary")},
        hygd_release={label: tail([r["Quality Score"] for r in official if r["Label"] == label], 2.)
                      for label in ("GON-", "GON+")},
        hygd_release_total=tail([r["Quality Score"] for r in official], 2.),
        fiqs_human_mos=tail([r["meanOpinionScore"] for r in fiqs], 20.),
        fiqs_by_original_level={str(level): tail(
            [r["meanOpinionScore"] for r in fiqs if r["qualityLevel"] == level], 20.)
            for level in sorted({r["qualityLevel"] for r in fiqs})},
        fiqs_validation_score_comparison={
            field: tail([r[field] for r in fiqs_val])
            for field in ("human_mos_normalized", "label", "prediction")},
        fiqs_validation_human_tail_recall={
            field: sum(float(r[field]) < .2 for r in fiqs_val
                       if float(r["human_mos_normalized"]) < .2)
            for field in ("label", "prediction")},
        eyeq_validation_human_grades=dict(Counter(r["eyeq_grade"] for r in eyeq_val if r["eyeq_grade"])),
        airogs_exploratory_by_label={
            label: {**tail([r["q"] for r in candidates if r["class"] == label]),
                    "screen_status": dict(Counter(r["screen_status"] for r in candidates if r["class"] == label))}
            for label in sorted({r["class"] for r in candidates})},
        paired_real={source: {field: tail([r[field] for r in pairs if r["source"] == source])
                              for field in ("q_low", "q_high")}
                     for source in sorted({r["source"] for r in pairs})},
        legacy_synthetic={variant: tail([r["q"] for r in legacy_pairs if r["variant"] == variant])
                          for variant in sorted({r["variant"] for r in legacy_pairs})},
    )
    write_json(OUTPUT / "inventories.json", result)


def review_worklist():
    candidates = read("data/external/airogs_candidate_5k/exploratory_scores.csv")
    manifest_hashes = {r["sha256"] for r in read("data/prepared/expanded_20261113/manifest.csv")}
    available = [r for r in candidates if r["class"] == "NRG"
                 and r["screen_status"] == "no_candidate_found"
                 and r["sha256"] not in manifest_hashes]
    available = list({r["sha256"]: r for r in available}.values())
    def order(row):
        return hashlib.sha256(f'20261116:{row["sha256"]}'.encode()).digest()

    low = [r for r in available if float(r["q"]) < .2]
    controls = [r for r in sorted(available, key=order) if float(r["q"]) >= .2][:30]
    path = OUTPUT / "airogs_blinded_quality_review.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "path", "overall_quality", "optic_disc_gradability",
                         "limiting_artifact", "review_notes"])
        for row in sorted(low + controls, key=order):
            source = Path(row["path"]).resolve()
            assert source.is_relative_to(ROOT / "data/external/airogs_candidate_5k")
            assert sha256(source) == row["sha256"]
            writer.writerow([row["sha256"][:16], source.relative_to(ROOT).as_posix(), "", "", "", ""])
    write_json(OUTPUT / "airogs_review_worklist_summary.json",
               dict(scored=len(candidates), eligible_for_review=len(available),
                    low_score_candidates=len(low), sampled_controls=len(controls),
                    provenance="NRG referral label; prior image screen and current exact-hash exclusion",
                    patient_overlap_review="pending", disease_label_admission="pending",
                    worklist_sha256=sha256(path)))


def provisional_review():
    directory = ROOT / "runs/followup_20261115/seed20261117/quota65_mixed50"
    update = json.loads((directory / "evaluation_summary.json").read_text())["selected_update"]
    predictions = read((directory / "development" / f"update_{update:06d}" /
                        "predictions.csv").relative_to(ROOT))
    quality = {r["sample_id"]: r for r in read("runs/followup_20261115/error_ranking/development_join.csv")
               if r["split"] == "boundary"}
    rows = [(r, quality[r["sample_id"]]) for r in predictions if r["condition"] == "original"]
    assert len(rows) == len(quality) == len({r["sample_id"] for r, _ in rows}) == 2053

    def remaining(data, budget, variants):
        reports = {}
        groups = sorted({r["related_group"] for r, _ in data})
        group_indices = {group: index for index, group in enumerate(groups)}
        rng = np.random.default_rng(20261116)
        resampled_groups = rng.integers(len(groups), size=(2000, len(groups)))
        bootstrap = np.array([np.bincount(row, minlength=len(groups)) for row in resampled_groups])
        baseline = None
        for name, classifier, priority in variants:
            def count(row):
                target = int(row[0]["label"])
                classification = classifier(row) > .5
                return int(target == 1 and not classification), int(target == 0 and classification)

            ranked = sorted(data, key=lambda row: (priority(row), row[0]["sample_id"]))
            counts = np.array([count(row) for row in ranked])
            missed, alarms = map(int, counts[budget:].sum(axis=0))
            total = len(data) - budget
            weights = bootstrap[:, [group_indices[r["related_group"]] for r, _ in ranked]]
            resampled_n = weights.sum(axis=1)
            resampled_budget = np.floor(.2 * resampled_n + .5).astype(int)
            cumulative = weights.cumsum(axis=1)
            automatic_weights = np.clip(cumulative - resampled_budget[:, None], 0, weights)
            residual = automatic_weights @ counts
            rate = residual.sum(axis=1) / (resampled_n - resampled_budget)
            if baseline is None:
                baseline = rate, residual[:, 0]
            reports[name] = dict(fn=missed, fp=alarms, errors=missed + alarms,
                                 automatic_n=total, error_rate=(missed + alarms) / total,
                                 before_review_fn=int(counts[:, 0].sum()), before_review_fp=int(counts[:, 1].sum()),
                                 delta_error_rate_vs_A_uncertainty_ci95=np.quantile(
                                     rate - baseline[0], [.025, .975]).tolist(),
                                 delta_fn_vs_A_uncertainty_ci95=np.quantile(
                                     residual[:, 0] - baseline[1], [.025, .975]).tolist())
        return reports

    p_a = lambda row: float(row[0]["prediction"])
    margin_a = lambda row: abs(p_a(row) - .5)
    q_floor = lambda row: (float(row[1]["q"]) >= .2, margin_a(row))
    eyeq_gate = lambda row: (row[1]["predicted_eyeq_grade"] != "2", margin_a(row))
    policies = [("A_uncertainty", p_a, margin_a), ("A_q_floor_first", p_a, q_floor),
                ("A_eyeq_reject_first", p_a, eyeq_gate)]
    known_b = {r["sample_id"]: r for r in read("data/audit/data_expansion_20261113/repaired_boundary_pq.csv")}
    common = [row for row in rows if row[0]["sample_id"] in known_b]
    assert len(common) == 1220
    assert all(known_b[r["sample_id"]]["label"] == r["label"]
               and known_b[r["sample_id"]]["source"] == r["source"] for r, _ in common)
    p_b = lambda row: float(known_b[row[0]["sample_id"]]["p_b"])
    b_policy = ("B_uncertainty", p_b, lambda row: abs(p_b(row) - .5))
    write_json(OUTPUT / "development_review_prototypes.json",
               dict(disease_threshold=.5, a_update=update, protected_test_inference=False,
                    cluster_bootstrap_repetitions=2000,
                    boundary=dict(n=len(rows), k=411, results=remaining(rows, 411, policies)),
                    shared_legacy_boundary=dict(n=len(common), k=244,
                        results=remaining(common, 244, policies + [b_policy])),
                    scope="fixed review budget development prototypes; no final boundary selected"))


def sample_development():
    rows = load_rows(COHORT / "manifest.csv", "glaucoma", "boundary", COHORT / "release.json")
    rows.sort(key=lambda r: hashlib.sha256(f'20261116:{r["sample_id"]}'.encode()).digest())
    selected, groups = [], set()
    for source in SOURCES:
        for label in ("0", "1"):
            stratum = [r for r in rows if r["source"] == source and r["label"] == label
                       and r["related_group"] not in groups]
            unique = {r["related_group"]: r for r in reversed(stratum)}
            chosen = list(reversed(unique.values()))[:12]
            assert len(chosen) == 12
            selected.extend(chosen)
            groups.update(r["related_group"] for r in chosen)
    return selected


def variants(row):
    path = (ROOT / "data" / row["path"]).resolve()
    assert path.is_relative_to(ROOT / "data") and sha256(path) == row["sha256"]
    with Image.open(path) as source:
        native = source.convert("RGB")
    side = max(native.size)
    image = ImageOps.pad(native, (side, side), color=0).resize(
        (TRAIN_SIZE, TRAIN_SIZE), Image.Resampling.BILINEAR)
    yield "native", native, image
    yield "train_input", image, image
    for family in FAMILIES:
        seed = int(hashlib.sha256(f'20261116:{row["sample_id"]}:{family}'.encode()).hexdigest()[:15], 16)
        changed = FundusDegradation(1., families=[family])(image, seed=seed)
        yield family, changed, changed
    diameter = retinal_context(image)["field_diameter_pixels"]
    blurred = image.filter(ImageFilter.GaussianBlur(5 * diameter / 512))
    dark = ImageEnhance.Brightness(image).enhance(.4)
    combined = ImageEnhance.Brightness(blurred).enhance(.4)
    for name, changed in (("strong_defocus", blurred), ("strong_underexposure", dark),
                          ("strong_combined", combined)):
        yield name, changed, changed


def model_at(config_path, ordinal=False):
    config = json.loads((ROOT / config_path).read_text())
    path = ROOT / config["run_dir"] / "best.pt"
    model = create_model(config["architecture"], ROOT / "models/checkpoints")
    if ordinal:
        model = QualityModel(model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), config["image_size"], sha256(path)


def summarize(rows):
    native = {r["sample_id"]: r for r in rows if r["condition"] == "native"}
    original = {r["sample_id"]: r for r in rows if r["condition"] == "train_input"}
    result = {}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        arm = [r for r in rows if r["condition"] == condition]
        scores = np.array([r["p"] for r in arm])
        labels = np.array([r["label"] for r in arm])
        scores_orig = np.array([original[r["sample_id"]]["p"] for r in arm])
        wrong = (scores > .5) != labels
        previous_wrong = (scores_orig > .5) != labels
        result[condition] = dict(
            q=tail([r["q"] for r in arm]), eyeq_q=tail([r["eyeq_q"] for r in arm]),
            eyeq_rejects=sum(r["eyeq_grade"] == 2 for r in arm),
            newly_below_q20=sum(r["q"] < .2 <= original[r["sample_id"]]["q"] for r in arm),
            newly_below_q20_by_label={str(label): sum(r["label"] == label and
                r["q"] < .2 <= original[r["sample_id"]]["q"] for r in arm) for label in (0, 1)},
            median_delta_q_vs_train=float(np.median([r["q"] - original[r["sample_id"]]["q"] for r in arm])),
            median_delta_q_vs_native=float(np.median([r["q"] - native[r["sample_id"]]["q"] for r in arm])),
            auc=float(roc_auc_score(labels, scores)), fn=int((wrong & (labels == 1)).sum()),
            fp=int((wrong & (labels == 0)).sum()), newly_wrong=int((wrong & ~previous_wrong).sum()),
            newly_correct=int((~wrong & previous_wrong).sum()),
            lowq_errors=int(sum(r["q"] < .2 and error for r, error in zip(arm, wrong))),
        )
    return result


def main():
    inventories()
    review_worklist()
    provisional_review()
    if "--inventories-only" in sys.argv:
        return
    torch.set_num_threads(2)
    student, size, q_digest = model_at("configs/formal_v3/quality_student.json")
    eyeq, eyeq_size, eyeq_digest = model_at("configs/eyeq_quality_v1.json", ordinal=True)
    disease, p_size, p_digest = model_at("configs/followup_20261115/seed20261117/quota65_mixed50.json")
    assert size == eyeq_size and p_size == TRAIN_SIZE
    quality_transform = ImageRows([], ROOT / "data", size, "quality").transform
    disease_transform = ImageRows([], ROOT / "data", p_size, "glaucoma").transform
    examples = sample_development()
    rows = []
    for start in range(0, len(examples), 2):
        batch = [(row, name, quality_transform(q_image), disease_transform(p_image))
                 for row in examples[start:start+2] for name, q_image, p_image in variants(row)]
        with torch.inference_mode():
            quality_inputs = torch.stack([item[2] for item in batch])
            scores = student(quality_inputs).flatten().tolist()
            q_eyeq, logits = eyeq(quality_inputs)
            grades = 2 - (logits > 0).sum(dim=1)
            probabilities = disease(torch.stack([item[3] for item in batch])).flatten().sigmoid()
        for (row, condition, _, _), q, eq, grade, p in zip(
                batch, scores, q_eyeq.tolist(), grades.tolist(), probabilities.tolist()):
            assert np.isfinite([q, eq, p]).all()
            rows.append(dict(sample_id=row["sample_id"], source=row["source"],
                             label=int(row["label"]), group=row["related_group"],
                             condition=condition, q=q, eyeq_q=eq, eyeq_grade=grade, p=p))
        if start % 20 == 0:
            print(f"Scored {start + 2}/{len(examples)} original groups", flush=True)
    reference = {r["sample_id"]: r for r in read("runs/followup_20261115/error_ranking/development_join.csv")}
    max_diff = max(abs(r["q"] - float(reference[r["sample_id"]]["q"]))
                   for r in rows if r["condition"] == "native")
    assert max_diff < 1e-3, max_diff
    path = OUTPUT / "paired_scores.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(original_groups=len(examples), rows=len(rows),
                  sources=list(SOURCES), sample_per_source_label=12, role="boundary", device="cpu",
                  paired_scores_sha256=sha256(path), q_checkpoint_sha256=q_digest,
                  eyeq_checkpoint_sha256=eyeq_digest, p_checkpoint_sha256=p_digest,
                  q_reference_max_abs_diff=max_diff, protected_test_inference=False,
                  results=summarize(rows))
    write_json(OUTPUT / "paired_summary.json", report)


if __name__ == "__main__":
    main()
