"""Summarize completed matched-defocus runs and archived teacher/student scores."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_expansion_results_20261113 import (
    align_rows, digest, group_weights, interval, point_metrics, read_csv, read_json,
    weighted_metrics, write_json,
)
from tools.run_followup_experiments_20261115 import RUN, SEEDS, SOURCES
from tools.run_matched_defocus_20261116 import ARM, STATUS

ARMS = ("quota65_clean", "quota65_mixed50", ARM, "quota65_defocus50")
CONDITIONS = ("original", "defocus_2_at512", "exposure_070")
OUT = ROOT / "docs/progress/2026-11-16"
REPEATS = 2000


def score_distribution(inputs):
    scores_path = ROOT / "runs/expanded_20261113/pq_quality_20261114/quality_scores.csv"
    proof = read_json(scores_path.with_name("verification.json"))
    assert digest(scores_path) == proof["scores_sha256"]
    scores = read_csv(scores_path)
    manifest_path = ROOT / "data/prepared/expanded_20261113/manifest.csv"
    manifest = {r["sample_id"]: r for r in read_csv(manifest_path)}
    teacher_path = ROOT / "data/audit/session_20261110/selected_validation/quality_student/validation_predictions.csv"
    teacher = {r["sha256"]: float(r["label"]) for r in read_csv(teacher_path)}
    roles, paired = {}, []
    for role in ("val", "boundary"):
        rows = [r for r in scores if r["split"] == role]
        assert all(manifest[r["sample_id"]]["split"] == role for r in rows)
        q = np.array([float(r["q"]) for r in rows])
        known = np.array([r["source"] in SOURCES[:3] for r in rows])
        roles[role] = dict(n=len(rows), below_020=int((q < .2).sum()),
                           below_010=int((q < .1).sum()), median=float(np.median(q)),
                           minimum=float(q.min()), original_sources_n=int(known.sum()),
                           original_sources_below_020=int(((q < .2) & known).sum()),
                           additional_sources_below_020=int(((q < .2) & ~known).sum()))
        for row in rows:
            image_hash = manifest[row["sample_id"]]["sha256"]
            if image_hash in teacher:
                assert role == "val"
                paired.append((teacher[image_hash], float(row["q"])))
    reference, student = np.array(paired).T
    low_t, low_s = reference < .2, student < .2
    for path in (scores_path, manifest_path, teacher_path):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    return dict(roles=roles, paired_validation=dict(
        n=len(paired), teacher_below_020=int(low_t.sum()), student_below_020=int(low_s.sum()),
        both_below_020=int((low_t & low_s).sum()), teacher_only=int((low_t & ~low_s).sum()),
        student_only=int((~low_t & low_s).sum()), median_abs_difference=float(np.median(abs(reference-student)))))


def read_runs(inputs):
    results = {}
    for mode in ("selected", "update2400"):
        for arm in ARMS:
            for seed in SEEDS:
                directory = RUN / f"seed{seed}" / arm
                training_path = directory / "run_summary.json"
                evaluation_path = directory / "evaluation_summary.json"
                train, evaluation = read_json(training_path), read_json(evaluation_path)
                assert train["optimizer_updates"] == 2400 and train["budget_completed"]
                update = evaluation["selected_update"] if mode == "selected" else 2400
                result = evaluation["checkpoints"][str(update)]
                path = directory / "development" / f"update_{update:06d}" / "predictions.csv"
                assert digest(path) == result["prediction_sha256"]
                assert not result["protected_test_inference"]
                rows = read_csv(path)
                assert {r["split"] for r in rows} == {"boundary"}
                for item in (training_path, evaluation_path, path):
                    inputs[item.relative_to(ROOT).as_posix()] = digest(item)
                results[mode, arm, seed] = result, train, rows
    return results


def main():
    status = read_json(STATUS)
    assert status["status"] == "completed" and len(status["completed"]) == len(SEEDS)
    inputs = {STATUS.relative_to(ROOT).as_posix(): digest(STATUS)}
    quality = score_distribution(inputs)
    runs = read_runs(inputs)
    quality_scores = {r["sample_id"]: float(r["q"]) for r in read_csv(
        ROOT / "runs/expanded_20261113/pq_quality_20261114/quality_scores.csv") if r["split"] == "boundary"}
    results, contrasts = [], {}
    for mode in ("selected", "update2400"):
        keys = [(mode, arm, seed) for arm in ARMS for seed in SEEDS]
        predictions = align_rows([runs[key][2] for key in keys])
        metrics = {key: {} for key in keys}
        contrasts[mode] = {}
        for condition in CONDITIONS:
            sets = [[r for r in rows if r["condition"] == condition] for rows in predictions]
            first = sets[0]
            assert len(first) == (2053 if condition == "original" else 384)
            assert set(r["sample_id"] for r in first).issubset(quality_scores)
            labels = np.array([int(r["label"]) for r in first])
            weights = group_weights([r["related_group"] for r in first], REPEATS, 20261116)
            bootstrap = {}
            for (_, arm, seed), rows in zip(keys, sets):
                values = np.array([float(r["prediction"]) for r in rows])
                bootstrap[arm, seed] = weighted_metrics(labels, values, weights)["auc"]
                metrics[mode, arm, seed][condition] = point_metrics(rows)
            contrasts[mode][condition] = {}
            for control in (arm for arm in ARMS if arm != ARM):
                deltas = [metrics[mode, ARM, seed][condition]["auc"]
                          - metrics[mode, control, seed][condition]["auc"] for seed in SEEDS]
                samples = np.mean([bootstrap[ARM, seed] - bootstrap[control, seed] for seed in SEEDS], axis=0)
                contrasts[mode][condition][control] = dict(
                    matched_minus_control=float(np.mean(deltas)), per_seed=list(deltas), **interval(samples))
        for key, rows in zip(keys, predictions):
            _, arm, seed = key
            result, training, _ = runs[key]
            originals = [r for r in rows if r["condition"] == "original"]
            low = [r for r in originals if quality_scores[r["sample_id"]] < .2]
            poor = [r for r in originals if r["real_poor_quality"] == "True"]
            results.append(dict(mode=mode, arm=arm, seed=seed, update=result["update"],
                                validation_macro_auc=result["validation_selection"]["selection_value"],
                                defocus_draws=training["degradation_family_draws"].get("defocus", 0),
                                total_draws=sum(training["degradation_family_draws"].values()),
                                metrics=metrics[key], low_score_originals=point_metrics(low),
                                annotated_poor_originals=point_metrics(poor)))
    write_json(OUT / "completed-matched-defocus.json", dict(
        status=status, quality=quality, runs=results, contrasts=contrasts,
        bootstrap_repeats=REPEATS, bootstrap_unit="related_group",
        protected_test_inference=False, input_sha256=inputs))


if __name__ == "__main__":
    main()
