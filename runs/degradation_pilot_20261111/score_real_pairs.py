"""Check the frozen quality model on two published same-eye photo datasets."""

import csv
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from training.data import ImageRows, sha256
from training.models import create_model


def list_pairs(archive):
    files = {
        p for entry in archive.namelist()
        if (p := Path(entry)).suffix.lower() in {".png", ".jpeg", ".jpg"} and "__MACOSX" not in p.parts
    }
    high_paths = sorted(p for p in files if p.parent.name == "gt" or p.stem.endswith("_good"))
    for high in high_paths:
        if high.parent.name == "gt":
            low = high.parent.parent / "input" / high.name
            pair_id = high.stem
        else:
            pair_id = high.stem.removesuffix("_good")
            matches = [p for p in files if p.stem == f"{pair_id}_bad"]
            assert len(matches) == 1, high
            low = matches[0]
        assert low in files, (low, high)
        yield pair_id, low.as_posix(), high.as_posix()


def score_archive(model, transform, path, count, source):
    rows = []
    with ZipFile(path) as archive:
        pairs = list(list_pairs(archive))
        assert len(pairs) == count
        with torch.inference_mode():
            for start in range(0, count, 8):
                batch = pairs[start:start + 8]
                tensors = []
                for _, *versions in batch:
                    for name in versions:
                        with archive.open(name) as stream, Image.open(stream) as image:
                            tensors.append(transform(image.convert("RGB")))
                scores = model(torch.stack(tensors).to("cuda")).flatten().cpu().numpy()
                assert np.isfinite(scores).all()
                for (pair_id, low, high), (q_low, q_high) in zip(batch, scores.reshape(-1, 2)):
                    rows.append({
                        "source": source, "pair_id": pair_id, "low_image": low, "high_image": high,
                        "q_low": float(q_low), "q_high": float(q_high), "delta_q_high_minus_low": float(q_high - q_low),
                    })
    delta = np.array([row["delta_q_high_minus_low"] for row in rows])
    summary = {
        "pairs": count, "higher_on_better_photo": int((delta > 0).sum()),
        "lower_on_better_photo": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
        "median_delta_q_high_minus_low": float(np.median(delta)),
        "archive_sha256": sha256(path),
    }
    return rows, summary


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads((ROOT / "configs/formal_v3/quality_student.json").read_text())
    path = ROOT / config["run_dir"] / "best.pt"
    expected = "96b2b32a0d9bc5b765e12f343d989ae1572ea5c4d4de3d4d198b93facb42a146"
    assert sha256(path) == expected
    transform = ImageRows([], ROOT / "data", config["image_size"], "quality", train=False).transform
    model = create_model(config["architecture"], ROOT / "models/checkpoints").to("cuda").eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
    datasets = {
        "Real Fundus": (ROOT / "data/external/real_fundus/Real_Fundus.zip", 120),
        "HRF same-eye": (ROOT / "data/external_quality/HRF_allQuality.zip", 18),
    }
    all_rows, summaries = [], {}
    for source, (archive, count) in datasets.items():
        rows, summaries[source] = score_archive(model, transform, archive, count, source)
        all_rows.extend(rows)
        print(source, summaries[source], flush=True)
    with (OUT / "real_pairs_quality.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summaries["checkpoint_sha256"] = expected
    (OUT / "real_pairs_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
