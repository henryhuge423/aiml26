"""Plot the two validation curves from the matched training runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPORT = Path(__file__).resolve().parents[1] / 'docs/progress/2026-11-18'


def main():
    result = json.loads((REPORT / 'backbone-results.json').read_text(encoding='utf-8'))
    models = [('efficientnet', 'EfficientNet-B0', '#20577e'),
              ('resnet50', 'ResNet-50', '#c35f31')]
    for lang, xlabel, headings in [
        ('en', 'Optimizer updates', ('AIROGS validation', 'Five-source validation mean')),
        ('zh', '参数更新次数', ('AIROGS 模型验证集', '五个来源的验证均值')),
    ]:
        font = Path('C:/Windows/Fonts/msyh.ttc')
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams['font.family'] = [font_manager.FontProperties(fname=font).get_name(), 'DejaVu Sans']
        fig, axes = plt.subplots(1, 2, figsize=(8.7, 2.65), sharey=True, layout='constrained')
        for axis, metric, heading in zip(axes, ('airogs_auc', 'five_source_macro_auc'), headings):
            for key, name, color in models:
                records = result['provenance'][key]['validation_curve']
                axis.plot([item['update'] for item in records], [item[metric] for item in records],
                          color=color, label=name, marker='o', markersize=3.6, linewidth=1.7)
            axis.set(title=heading, xlabel=xlabel, xticks=[300, 900, 1500, 2100, 2400])
            axis.grid(axis='y', color='#dbe4ea')
            axis.set_axisbelow(True)
            axis.spines[['right', 'top']].set_visible(False)
        axes[0].set_ylabel('AUROC')
        axes[0].legend(loc='lower right', fontsize=8, frameon=False)
        fig.savefig(REPORT / f'backbone-validation.{lang}.png', dpi=200, facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    main()
