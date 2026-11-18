"""Render the paired quality-score distributions for the bilingual report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/progress/2026-11-16"
LABELS = {
    "en": [
        "Native", "320 px baseline", "Mild illumination", "Mild defocus",
        "Mild scatter", "Strong defocus", "Strong underexposure", "Strong defocus + underexposure",
    ],
    "zh": [
        "原图直接评分", "320 像素对照", "温和照明", "温和失焦",
        "温和散射", "强失焦", "强欠曝", "强失焦加欠曝",
    ],
}


def main():
    summary = json.loads((REPORT / "paired-summary.json").read_text(encoding="utf-8"))
    arms = list(summary["results"].values())
    for lang, labels in LABELS.items():
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        font = FontProperties(fname=font_path) if lang == "zh" and font_path.exists() else None
        fig, ax = plt.subplots(figsize=(8.3, 3.45), layout="constrained")
        for i, arm in enumerate(arms):
            scores = arm["q"]
            color = "#68768a" if i < 2 else "#196878" if i < 5 else "#a5462c"
            ax.hlines(i, scores["p10"], scores["p90"], color=color, linewidth=2.3, alpha=.8)
            ax.scatter(scores["median"], i, color=color, s=32, zorder=3)
        ax.axvline(.2, color="#344057", linestyle="--", linewidth=1.1)
        ax.set_xlim(0, 1)
        ax.set_yticks(range(len(arms)), labels, fontproperties=font, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("FTHNet 蒸馏质量分数 q" if lang == "zh" else "FTHNet student quality score q",
                      fontproperties=font, fontsize=10)
        ax.grid(axis="x", color="#e4e8ed")
        ax.tick_params(axis="both", length=0, pad=7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c3ccd6")
        fig.savefig(REPORT / f"paired-quality.{lang}.png", dpi=205)
        plt.close(fig)


if __name__ == "__main__":
    main()
