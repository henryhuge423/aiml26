"""Plot the frozen audit disposition and admitted source composition."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
POOL = ROOT / 'data/prepared/expansion_20261113/release.json'
AUDIT = ROOT / 'data/audit/data_expansion_20261113/local_summary.json'
LABELS = {'airogs_original_rg': 'AIROGS RG', 'beh': 'BEH', 'chaksu': 'Chákṣu',
          'crfo': 'CRFO', 'fives': 'FIVES', 'fundus_avseg': 'Fundus-AVSeg',
          'grape': 'GRAPE', 'hrf': 'HRF', 'keh': 'KEH', 'lag_part1': 'LAG',
          'les_av': 'LES-AV', 'mshf': 'MSHF', 'papila': 'PAPILA'}

def main():
    pool = json.loads(POOL.read_text(encoding='utf-8'))
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    assert pool['images'] == 9070 and pool['positive_images'] == 3446
    assert audit['new_images'] == 13176 and pool['auxiliary_suspect_images'] == 204
    names = sorted(pool['sources'], key=lambda s: pool['sources'][s]['images'], reverse=True)
    positives = np.array([pool['sources'][s]['positives'] for s in names])
    totals = np.array([pool['sources'][s]['images'] for s in names])
    assert totals.sum() == 9070 and positives.sum() == 3446
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), gridspec_kw={'width_ratios': [1.45, 1]}, layout='constrained')
    y = np.arange(len(names))
    axes[0].barh(y, positives, color='#d87935', label='Positive labels')
    axes[0].barh(y, totals - positives, left=positives, color='#4c83b6', label='Negative labels')
    axes[0].set_yticks(y, [LABELS[s] for s in names])
    axes[0].invert_yaxis()
    for i, n in enumerate(totals): axes[0].text(n + 50, i, f'{n:,}', va='center', fontsize=9)
    axes[0].set(xlim=(0, 5750), xlabel='Admitted disease-labeled images', title='Source composition of the 9,070-image pool')
    axes[0].legend(loc='lower right', frameon=False, fontsize=9)
    axes[0].grid(axis='x', alpha=.2)
    axes[0].set_axisbelow(True)
    values = [9070, 204, 13176 - 9070 - 204]
    axes[1].barh([0, 1, 2], values, color=['#4c83b6', '#8a67a5', '#adb5bd'], height=.5)
    axes[1].set_yticks([0, 1, 2], ['Disease pool', 'Auxiliary suspects', 'Excluded'])
    axes[1].invert_yaxis()
    for i, n in enumerate(values):
        axes[1].text(n + 150, i, f'{n:,} ({n/13176:.1%})', va='center', fontsize=10)
    axes[1].set(xlim=(0, 13000), ylim=(2.6, -.6), xlabel='Candidate photographs', title='Disposition of all 13,176 candidates')
    axes[1].grid(axis='x', alpha=.2)
    axes[1].set_axisbelow(True)
    axes[1].text(.0, -.12, 'Counts refer to photographs; suspects are stored separately.', transform=axes[1].transAxes, fontsize=8)
    fig.savefig(HERE / 'admission-and-sources.png', dpi=180)
    plt.close(fig)
    record = {'inputs_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (POOL, AUDIT)},
              'candidate_images': 13176, 'disease_pool_images': 9070, 'positive_images': 3446,
              'auxiliary_suspects': 204, 'excluded_images': values[2], 'sources': pool['sources']}
    (HERE / 'admission-chart-data.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k not in ('inputs_sha256', 'sources')}, indent=2))

if __name__ == '__main__': main()
