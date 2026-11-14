"""Build the bilingual November 13 overview from audited experiment summaries."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/progress/2026-11-13'
BUILD = ROOT / '.build/progress-overview-20261113'
ARMS = ('original_clean', 'expanded_clean', 'expanded_degradation30', 'expanded_degradation50')

def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    cohort_path = ROOT / 'data/prepared/expanded_20261113/cohort_verification.json'
    cohort = json.loads(cohort_path.read_text(encoding='utf-8'))
    comparison_path = ROOT / 'runs/expanded_20261113/budget_end_comparison.json'
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    assert cohort['unique_paths_file_hash_and_full_RGB_verified'] == 24491
    for arm in ARMS:
        summary = json.loads((ROOT / f'runs/expanded_20261113/{arm}/run_summary.json').read_text(encoding='utf-8'))
        assert summary['budget_completed'] and summary['optimizer_updates'] == 1200
    colors = ['#4a5568', '#2378a4', '#24866b', '#be7c25']
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'Arial'], 'axes.unicode_minus': False,
                         'font.size': 10, 'savefig.facecolor': 'white'})
    for lang in ('zh', 'en'):
        zh = lang == 'zh'
        fig, ax = plt.subplots(figsize=(9.3, 3.5))
        roles = ['训练', '模型验证', '规则开发', '内部及新增测试'] if zh else ['Training', 'Model validation', 'Rule development', 'Internal / added tests']
        original, additions = np.array([10338, 1188, 1220, 1233]), np.array([5142, 858, 833, 2237])
        x = np.arange(4)
        ax.barh(x, original, color=colors[0], label='原数据（修正后）' if zh else 'Retained original data')
        ax.barh(x, additions, left=original, color=colors[1], label='本轮新增数据' if zh else 'Added data')
        for i, (a, b) in enumerate(zip(original, additions)):
            ax.text(a / 2, i, f'{a:,}', color='white', ha='center', va='center')
            ax.text(a + b / 2, i, f'{b:,}', color='white', ha='center', va='center')
            ax.text(a + b + 160, i, f'{a+b:,}', va='center')
        ax.set_yticks(x, roles)
        ax.invert_yaxis()
        ax.set_xlim(0, 18000)
        ax.set_xlabel('图像数' if zh else 'Images')
        ax.set_title('原数据与新增数据按用途分配' if zh else 'Original and added images by role')
        ax.legend(loc='lower right', frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f'cohort-allocation.{lang}.png', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.5))
        sources = ['airogs-light-v2', 'DATASET_101/ODIR-5K', 'lag_part1', 'keh']
        labels = ['原数据', '扩展', '扩展+30%', '扩展+50%'] if zh else ['Original', 'Expanded', 'Expanded+30%', 'Expanded+50%']
        x, width = np.arange(4), .18
        for i, (arm, label, color) in enumerate(zip(ARMS, labels, colors)):
            values = [comparison['arms'][arm]['sets']['boundary/original']['per_source'][s]['auc'] for s in sources]
            axes[0].bar(x + (i - 1.5) * width, values, width, label=label, color=color)
        axes[0].set_xticks(x, ['AIROGS', 'ODIR', 'LAG', 'KEH'])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('AUROC')
        axes[0].set_title('规则开发集原图' if zh else 'Original rule-development images')
        misses, false_positives = [55, 18, 36, 28], [10, 68, 23, 19]
        axes[1].bar(x - .17, np.array(misses) / 146 * 100, .34, color='#b65745', label='漏诊率（146张阳性）' if zh else 'Miss rate (146 positives)')
        axes[1].bar(x + .17, np.array(false_positives) / 238 * 100, .34, color='#2378a4', label='误报率（238张阴性）' if zh else 'False-positive rate (238 negatives)')
        axes[1].set_xticks(x, ['原数据', '扩展', '30%', '50%'] if zh else ['Original', 'Expanded', '30%', '50%'])
        axes[1].set_ylabel('比例（%）' if zh else 'Rate (%)')
        axes[1].set_title('同一384张开发图的合成失焦' if zh else 'Synthetic defocus of the same 384 images')
        axes[1].set_ylim(0, 46)
        for ax in axes:
            ax.legend(fontsize=8, loc='lower left' if ax is axes[0] else 'upper right', frameon=False)
            ax.grid(axis='y', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f'experiment-overview.{lang}.png', dpi=180)
        plt.close(fig)
    return {'cohort_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            'comparison_sha256': hashlib.sha256(comparison_path.read_bytes()).hexdigest()}

def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    inputs = charts()
    result = {'inputs': inputs, 'languages': {}}
    for lang in ('zh', 'en'):
        stem = f'progress-overview.{lang}'
        tex = OUT / f'{stem}.tex'
        subprocess.run(['pandoc', str(OUT / f'{stem}.md'), '-s', '-t', 'latex', '-o', str(tex),
                        '-V', 'mainfont=Arial', '-V', 'CJKmainfont=Microsoft YaHei', '-V', 'monofont=Consolas',
                        '-V', 'fontsize=10pt', '-V', 'geometry:margin=0.7in', '-V', 'colorlinks=true',
                        '-V', 'urlcolor=blue'], cwd=OUT, check=True, capture_output=True)
        text = tex.read_text(encoding='utf-8').replace('\\begin{figure}', '\\begin{figure}[H]')
        preamble = '\\usepackage{float}\n\\usepackage{needspace}\n\\Urlmuskip=0mu plus 2mu\\relax\n'
        if lang == 'zh':
            preamble += '\\renewcommand{\\figurename}{图}\n'
        text = text.replace('\\begin{longtable}', '\\Needspace{8\\baselineskip}\n\\begin{longtable}')
        tex.write_text(text.replace('\\begin{document}', preamble + '\\begin{document}'), encoding='utf-8')
        for index in range(2):
            process = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                                      f'-output-directory={BUILD}', str(tex)], cwd=OUT, capture_output=True)
            (BUILD / f'{stem}-build-{index}.log').write_bytes(process.stdout + process.stderr)
            process.check_returncode()
        shutil.copyfile(BUILD / f'{stem}.pdf', OUT / f'{stem}.pdf')
        subprocess.run(['pdftotext', '-layout', str(OUT / f'{stem}.pdf'), str(BUILD / f'{stem}.txt')], check=True)
        subprocess.run(['pdftoppm', '-scale-to', '1500', '-png', str(OUT / f'{stem}.pdf'), str(BUILD / f'page-{lang}')], check=True, capture_output=True)
        result['languages'][lang] = {'pages': (BUILD / f'{stem}.txt').read_text(encoding='utf-8').count('\f'),
                                     'pdf_sha256': hashlib.sha256((OUT / f'{stem}.pdf').read_bytes()).hexdigest()}
    (OUT / 'overview-build.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    build()
