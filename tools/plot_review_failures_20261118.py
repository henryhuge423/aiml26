"""Draw bilingual figures and build a local contact sheet from verified images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fundus_degradation import retinal_context

OUTPUT = ROOT / 'docs/progress/2026-11-18'
LOCAL = ROOT / 'runs/review_failures_20261118'
MANIFEST = ROOT / 'data/prepared/expanded_20261113/manifest.csv'
REAL = ROOT / 'data/external/real_fundus/Real_Fundus.zip'
BLUE = '#1b5d82'
ORANGE = '#b65229'
RED = '#a62f48'
GRAY = '#aab8c1'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def axes_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(alpha=.22, color='#7e98a8')


def disagreement(dev, e_cut, r_cut, lang, labels):
    fig, ax = plt.subplots(figsize=(7.0, 3.2), layout='constrained')
    positive = dev[dev.label.eq(1)]
    call_e = positive.efficientnet > e_cut
    call_r = positive.resnet50 > r_cut
    categories = [
        (call_e, GRAY, 'o', labels[0]),
        (~call_e & call_r, ORANGE, '^', labels[1]),
        (~call_e & ~call_r, BLUE, 's', labels[2]),
    ]
    for mask, color, marker, label in categories:
        ax.scatter(positive.loc[mask,'efficientnet'], positive.loc[mask,'resnet50'],
                   c=color, marker=marker, s=19, alpha=.83, edgecolors='none', label=f'{label} ({mask.sum()})')
    ax.axvline(e_cut, linewidth=1, color='#4c5b66', linestyle='--')
    ax.axhline(r_cut, linewidth=1, color='#4c5b66', linestyle='--')
    ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xlabel=labels[3], ylabel=labels[4])
    axes_style(ax)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, .53), frameon=False, borderaxespad=.9)
    fig.savefig(OUTPUT / f'review-model-disagreement.{lang}.png', dpi=195, facecolor='white')
    plt.close(fig)


def focus(dev, pairs, cut, lang, labels):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.1), layout='constrained')
    positive = dev.label.eq(1)
    scores = dev[['original_repeat','blur_2','blur_5']]
    miss = scores.loc[positive].le(cut).mean(axis=0) * 100
    alarm = scores.loc[~positive].gt(cut).mean(axis=0) * 100
    for metric, color, label in [(miss, RED, labels[4]), (alarm, BLUE, labels[5])]:
        left.plot(range(3), metric, marker='o', color=color, linewidth=1.8, label=label)
        for x, y in enumerate(metric):
            left.annotate(f'{y:.1f}%', (x, y), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
    upper = max(26, max(miss.max(), alarm.max()) + 9)
    left.set(xticks=range(3), xticklabels=labels[0:3], title=labels[3], ylabel=labels[6], ylim=(0, upper))
    left.legend(fontsize=8, loc='upper left', frameon=False)
    right.scatter(pairs.p_high, pairs.p_low, c=BLUE, s=18, alpha=.59, edgecolors='none')
    right.plot([0, 1], [0, 1], linewidth=1, color='#4c5b66', linestyle='--')
    right.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), title=labels[7], xlabel=labels[8], ylabel=labels[9])
    for ax in (left,right):
        axes_style(ax)
    fig.savefig(OUTPUT / f'review-paired-focus.{lang}.png', dpi=195, facecolor='white')
    plt.close(fig)


def tile(axis, image, caption):
    axis.imshow(image)
    axis.set_title(caption, loc='left', fontsize=9, color='#1c3949', pad=7)
    axis.axis('off')


def example_sheet(dev, pairs, e_cut, r_cut):
    eligible = dev.label.eq(1) & dev.efficientnet.le(e_cut) & ~dev.base_review
    groups = [
        dev[eligible & dev.resnet50.gt(r_cut) & dev.alternate_review].sort_values('q_eyeq', ascending=False),
        dev[eligible & dev.resnet50.le(r_cut) & ~dev.alternate_review].sort_values('q_eyeq', ascending=False),
    ]
    manifest = pd.read_csv(MANIFEST, usecols=['sample_id','path','sha256','split']).set_index('sample_id')
    fig, axs = plt.subplots(2, 4, figsize=(10.4, 5.6), layout='constrained')
    fig.suptitle('Eye Sentinel · four development examples', fontsize=15, color='#193749', fontweight='bold')
    for row, subset, heading in zip(range(2), groups, ('A · Added to review', 'B · Still outside review')):
        record = subset.iloc[0]
        m = manifest.loc[record.sample_id]
        path = ROOT / 'data' / m.path
        assert m.split == 'boundary' and sha256(path) == m.sha256
        with Image.open(path) as original:
            side = max(original.size)
            square = ImageOps.pad(original.convert('RGB'), (side,side), color=0).resize((320,320), Image.Resampling.BILINEAR)
        tile(axs[row,0], square, f'{heading}\nOriginal E {record.efficientnet:.2f} · R {record.resnet50:.2f} · q {record.q_eyeq:.2f}')
        sigma = 5 * retinal_context(square)['field_diameter_pixels'] / 512
        tile(axs[row,1], square.filter(ImageFilter.GaussianBlur(sigma)),
             f'Same photo, synthetic blur\nE {record.blur_5:.2f}')
    rank = (pairs.p_low-pairs.p_high).sort_values()
    with ZipFile(REAL) as archive:
        for row, (fraction, letter) in enumerate(((.10,'C'),(.90,'D'))):
            record = pairs.loc[rank.index[round(fraction*(len(rank)-1))]]
            for j, (suffix, name) in enumerate([('high','Clearer photo'),('low','Poorer photo')], 2):
                with archive.open(record[f'{suffix}_image']) as stream, Image.open(stream) as image:
                    tile(axs[row,j], image.convert('RGB'),
                         f'{letter} · {name}\nE {record[f"p_{suffix}"]:.3f}')
    fig.savefig(LOCAL/'fundus-examples.png', dpi=190, facecolor='white')
    plt.close(fig)


def main():
    result = json.loads((OUTPUT/'review-failures-results.json').read_text(encoding='utf-8'))
    path = LOCAL/'airogs-development.csv'
    pairs_path = LOCAL/'real-fundus-development.csv'
    assert sha256(path) == result['provenance']['airogs_case_table_sha256']
    assert sha256(pairs_path) == result['provenance']['real_pairs_table_sha256']
    assert sha256(REAL) == result['real_fundus']['archive_sha256']
    dev, pairs = pd.read_csv(path), pd.read_csv(pairs_path)
    e, r = [result['ranking']['thresholds'][key] for key in ('efficientnet','resnet50')]
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans']
    for lang, labels in {
        'zh': ['E 判阳', '仅 R 判阳', '两者都判阴', 'EfficientNet-B0 疾病分数', 'ResNet-50 疾病分数'],
        'en': ['E refers', 'Only R refers', 'Both miss', 'EfficientNet-B0 score', 'ResNet-50 score'],
    }.items():
        disagreement(dev, e, r, lang, labels)
    for lang, labels in {
        'zh': ['原图','轻失焦','强失焦','同一批 AIROGS 照片，模拟失焦',
               '漏诊','误报','占同标签照片的比例 (%)','120 对真实同眼照片','较清晰照片的疾病分数','较差照片的疾病分数'],
        'en': ['Original','Mild blur','Strong blur','AIROGS · same photos, simulated blur',
               'Misses','False positives','Share of each reference class (%)','120 real same-eye pairs','Score on the clearer photo','Score on the poorer photo'],
    }.items():
        focus(dev, pairs, e, lang, labels)
    example_sheet(dev,pairs,e,r)


if __name__ == '__main__':
    main()
