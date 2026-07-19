"""Render the manuscript's loss-indexed-confidence schematic as vector PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


INK = "#202733"
BLUE = "#004c99"
PALE_BLUE = "#e8f1f8"
PALE_GOLD = "#fbf2d5"
GRAY = "#66717e"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/Figures/loss_indexed_framework.pdf"),
    )
    return parser.parse_args(argv)


def _mask_axis(figure, bounds, values, title, *, probability=False):
    axis = figure.add_axes(bounds)
    axis.imshow(
        values,
        cmap="Blues" if probability else "binary",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7, color=INK, pad=2)
    for spine in axis.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.5)
    return axis


def _arrow(figure, start, end, *, color=GRAY, bend=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=figure.transFigure,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=0.8,
        color=color,
        connectionstyle=f"arc3,rad={bend}",
        clip_on=False,
    )
    figure.add_artist(arrow)


def _text_box(figure, x, y, text, *, width_color=PALE_BLUE, fontsize=7):
    figure.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": width_color,
            "edgecolor": BLUE,
            "linewidth": 0.7,
        },
    )


def render(output):
    probability = np.array(
        [
            [0.02, 0.05, 0.10, 0.06, 0.02],
            [0.05, 0.28, 0.58, 0.31, 0.07],
            [0.11, 0.62, 0.96, 0.73, 0.16],
            [0.04, 0.34, 0.77, 0.49, 0.08],
            [0.01, 0.06, 0.18, 0.07, 0.02],
        ]
    )
    gamma = 0.5
    thresholds = (0.2, 0.5, 0.8)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.1, 1.65), facecolor="white")

    _mask_axis(figure, [0.015, 0.31, 0.105, 0.46], probability, r"probability map $p(x)$", probability=True)
    figure.text(0.067, 0.18, "one model output", ha="center", color=GRAY, fontsize=6.3)

    _mask_axis(
        figure,
        [0.185, 0.52, 0.075, 0.33],
        probability >= gamma,
        r"deployed action $\widehat{Y}$",
    )
    figure.text(
        0.222,
        0.43,
        r"fixed once at $\gamma$",
        ha="center",
        color=GRAY,
        fontsize=6.3,
    )

    level_x = (0.17, 0.245, 0.32)
    for x, threshold in zip(level_x, thresholds):
        _mask_axis(
            figure,
            [x, 0.08, 0.06, 0.27],
            probability >= threshold,
            rf"$Y_{{{threshold:.1f}}}$",
        )
    figure.text(
        0.275,
        0.015,
        r"nested level sets at integration nodes $t_m$",
        ha="center",
        color=GRAY,
        fontsize=6.3,
    )

    _text_box(
        figure,
        0.485,
        0.51,
        r"same deployed action" "\n" r"$\ell_m=L_x(Y_{t_m},\widehat{Y})$",
        width_color=PALE_GOLD,
        fontsize=6.6,
    )
    _text_box(
        figure,
        0.665,
        0.51,
        r"integrate the chosen loss" "\n" r"$r_{L,M}=\sum_m w_m\ell_m$" "\n" r"$C_{L,M}=-r_{L,M}$",
        fontsize=6.6,
    )

    curve = figure.add_axes([0.81, 0.24, 0.17, 0.54])
    coverage = np.linspace(0.04, 1.0, 80)
    risk = 0.13 + 0.36 * coverage**1.55
    curve.plot(coverage, risk, color=BLUE, linewidth=1.25)
    curve.fill_between(coverage, 0, risk, color=PALE_BLUE)
    curve.set_xlim(0, 1)
    curve.set_ylim(0, 0.55)
    curve.set_xticks([0, 1])
    curve.set_yticks([0, 0.5])
    curve.tick_params(labelsize=5.8, length=2, pad=1)
    curve.set_xlabel("coverage", fontsize=6.3, labelpad=0)
    curve.set_ylabel(r"risk $R_L$", fontsize=6.3, labelpad=0)
    curve.set_title(r"rank by $C_{L,M}$", fontsize=7, color=INK, pad=2)
    curve.text(0.58, 0.17, r"$\mathrm{AURC}_L$", color=BLUE, fontsize=7)
    for spine in curve.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.5)

    _arrow(figure, (0.12, 0.55), (0.178, 0.68), color=BLUE)
    _arrow(figure, (0.12, 0.49), (0.16, 0.24), color=BLUE)
    _arrow(figure, (0.265, 0.68), (0.414, 0.58), bend=-0.04)
    _arrow(figure, (0.38, 0.22), (0.414, 0.43), bend=0.08)
    _arrow(figure, (0.555, 0.51), (0.594, 0.51), color=BLUE)
    _arrow(figure, (0.738, 0.60), (0.808, 0.70), color=BLUE, bend=-0.08)

    figure.text(
        0.59,
        0.95,
        r"changing $L$ changes confidence, not the deployed mask",
        ha="center",
        color=BLUE,
        fontsize=7,
        weight="bold",
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.01,
        metadata={
            "Title": "Loss-indexed confidence framework",
            "Author": "Anonymous",
            "Creator": "scripts/make_method_figure.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    render(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
