"""Run the dose-matched defocus arm on the three existing training seeds."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_expansion_experiments_20261113 import child, write_json
from tools.run_followup_experiments_20261115 import CONFIGS, RUN, SEEDS

ARM = "quota65_defocus_matched"
STATUS = RUN / "matched_defocus" / "status.json"


def main():
    status = dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  completed=[], total_runs=len(SEEDS), status="training",
                  protected_test_inference=False)
    try:
        for seed in SEEDS:
            name = f"seed{seed}/{ARM}"
            directory = RUN / name
            config = json.loads((CONFIGS / f"seed{seed}/quota65_defocus50.json").read_text())
            config["run_dir"] = directory.relative_to(ROOT).as_posix()
            config["degradation"]["probability"] = 0.5 / 3
            path = CONFIGS / f"{name}.json"
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError(f"configuration changed: {path}")
            write_json(path, config)
            status.update(current_run=name, status="training")
            write_json(STATUS, status)
            log = STATUS.parent / f"{seed}-train.log"
            summary = directory / "run_summary.json"
            if not (summary.exists() and json.loads(summary.read_text())["budget_completed"]):
                arguments = ["tools/train_model.py", "--config", str(path)]
                if (directory / "last.pt").exists():
                    arguments += ["--resume", str(directory / "last.pt")]
                child(arguments, log)
            status["status"] = "evaluating"
            write_json(STATUS, status)
            child(["tools/evaluate_followup_20261115.py", "--run", name],
                  STATUS.parent / f"{seed}-evaluation.log")
            status["completed"].append(name)
        status.update(status="completed", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(STATUS, status)
    except BaseException as error:
        status.update(status="failed", error=str(error))
        write_json(STATUS, status)
        raise


if __name__ == "__main__":
    main()
