"""Diagnose checkpoint selection using the same budget-end weight in each arm."""
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_expansion_experiments_20261113 import ARMS,RUN,child,write_json


def main():
    status_path=RUN/'pipeline/budget_end_status.json'
    status=dict(status='waiting_primary_driver',completed=[])
    write_json(status_path,status)
    while True:
        primary=json.loads((RUN/'pipeline/status.json').read_text())
        if primary['status']=='failed':raise RuntimeError('primary experiment driver failed')
        if primary['status']=='completed':break
        time.sleep(15)
    for arm in ARMS:
        status.update(status='evaluating',current_arm=arm);write_json(status_path,status)
        if not (RUN/arm/'last_development_metrics.json').exists():
            child(['tools/evaluate_expansion_20261113.py','--arm',arm,'--checkpoint','last'],RUN/f'pipeline/{arm}-budget-end.log')
        report=json.loads((RUN/arm/'last_development_metrics.json').read_text())
        if report['selected_successful_updates']!=1200:raise ValueError('budget-end checkpoint updates differ')
        status['completed'].append(arm);write_json(status_path,status)
    reports={a:json.loads((RUN/a/'last_development_metrics.json').read_text()) for a in ARMS}
    write_json(RUN/'budget_end_comparison.json',dict(interpretation='additional development diagnostic; does not change the predeclared selected checkpoints',
        successful_updates_each=1200,protected_test_inference=False,arms=reports))
    status.update(status='completed',finished_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json(status_path,status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
