"""Render the paired pilot scores and one source image with its variants."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
COLORS = {0: "#276678", 1: "#ad583d"}


def position(score):
    probability = np.clip(score, 0.0001, 0.9999)
    return np.log10(probability / (1 - probability))


def main():
    scores = pd.read_csv(OUT / "paired_scores.csv")
    cases = (
        ("original", "原图"),
        ("blur_strong", "较强模糊"),
        ("dark_strong", "较强暗光"),
    )
    ticks = np.array([.0001, .01, .1, .5, .9, .99, .9999])
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.78, wspace=.13)
    fig.suptitle("同一批眼底照片在退化前后的分数分布", x=.07, y=.96,
                 ha="left", fontproperties=FONT, fontsize=20, color="#173138")
    axes[0].set_ylabel("质量模型原始分数 q", fontproperties=FONT, fontsize=11)
    for ax, (variant, title) in zip(axes, cases):
        group = scores.loc[scores.variant.eq(variant)]
        ax.set_title(title, loc="left", fontproperties=FONT, fontsize=13, pad=10)
        ax.axvline(0, linestyle="--", lw=1.1, color="#7c8589", zorder=1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e1e7e7", lw=.8)
        for label, subset in group.groupby("label"):
            disagrees = subset.p_a.ge(.5).ne(subset.label.astype(bool))
            for mask, marker, size, alpha in (
                (~disagrees, "o", 17, .48),
                (disagrees, "x", 28, .86),
            ):
                ax.scatter(position(subset.loc[mask, "p_a"]), subset.loc[mask, "q"],
                           c=COLORS[label], marker=marker, s=size, alpha=alpha,
                           linewidths=1 if marker == "x" else 0, zorder=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#aebcbc")
        ax.set_xticks(position(ticks), [f"{t:g}" for t in ticks], fontsize=8)
        ax.set_xlim(-4.23, 4.23)
    fig.supxlabel("青光眼模型 A 的分数 p（非线性刻度，展开接近 0 和 1 的分数）",
                  y=.12, fontproperties=FONT, fontsize=11)
    fig.text(.07, .038, "蓝色：原标签为阴性", fontproperties=FONT, fontsize=10, color=COLORS[0])
    fig.text(.25, .038, "棕色：原标签为阳性", fontproperties=FONT, fontsize=10, color=COLORS[1])
    fig.text(.43, .038, "×：以 p = 0.5 展示时与原标签不一致",
             fontproperties=FONT, fontsize=10, color="#465b61")
    fig.savefig(OUT / "distribution.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    fig.subplots_adjust(left=.02, right=.98, top=.82, bottom=.05, wspace=.07)
    fig.suptitle("同一张照片的三种版本", x=.02, y=.98, ha="left",
                 fontproperties=FONT, fontsize=16, color="#173138")
    for ax, (variant, title) in zip(axes, cases):
        with Image.open(OUT / f"example_{variant}.png") as image:
            ax.imshow(image)
        ax.set_title(title, loc="left", fontproperties=FONT, fontsize=11, pad=7)
        ax.set_axis_off()
    fig.savefig(OUT / "example.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
