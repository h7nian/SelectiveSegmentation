"""Render a compact, descriptive score-geometry diagnostic table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
CONDITIONS = ("clipseg-target", "deeplabv3-target")
LABELS = {
    "pet": "Pet",
    "kvasir": "Kvasir",
    "fives": "FIVES",
    "isic": "ISIC",
    "tn3k": "TN3K",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembled-root", default="outputs/binary_midpoint_main_v2/assembled"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _spearman(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Spearman inputs must be equal nontrivial vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite")
    value = np.corrcoef(
        rankdata(left, method="average"), rankdata(right, method="average")
    )[0, 1]
    if not np.isfinite(value):
        raise ValueError("Spearman correlation is undefined")
    return float(value)


def condition_diagnostics(rows: list[dict]) -> dict[str, float]:
    required = {
        "risk_dice",
        "risk_nhd",
        "confidence_dice_m32",
        "confidence_dice_exact",
        "confidence_nhd_m32",
        "confidence_sdc",
        "confidence_foreground_entropy",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("assembled rows lack required mechanism fields")
    def vector(field):
        return np.asarray([row[field] for row in rows], dtype=float)
    dice = vector("confidence_dice_m32")
    dice_safety = -vector("risk_dice")
    nhd_safety = -vector("risk_nhd")
    return {
        "dice_fg_redundancy": _spearman(
            dice, vector("confidence_foreground_entropy")
        ),
        "dice_minus_sdc": _spearman(dice, dice_safety)
        - _spearman(vector("confidence_sdc"), dice_safety),
        "nhd_spatial_increment": _spearman(
            vector("confidence_nhd_m32"), nhd_safety
        )
        - _spearman(dice, nhd_safety),
        "exact_m32_difference": abs(
            _spearman(vector("confidence_dice_exact"), dice_safety)
            - _spearman(dice, dice_safety)
        ),
    }


def _load_condition(root: Path, dataset: str, condition: str) -> list[dict]:
    paths = sorted((root / dataset / condition).glob("*/records.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one assembled record file for {dataset}/{condition}")
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def render_table(values: dict[tuple[str, str], dict[str, float]]) -> str:
    expected = {(dataset, condition) for dataset in DATASETS for condition in CONDITIONS}
    if set(values) != expected:
        raise ValueError("mechanism table requires exactly ten target conditions")
    rows = (
        ("dice_fg_redundancy", r"Dice--FGEnt score redundancy"),
        ("dice_minus_sdc", r"Dice minus SDC association"),
        ("nhd_spatial_increment", r"nHD minus Dice association"),
        ("exact_m32_difference", r"Exact--M32 absolute difference"),
    )
    output = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Descriptive mechanism clues from the ten target conditions. Each cell is CLIP-T/DL-T and every association is Spearman's $\rho$. Row 1 is $\rho(C_{\rm Dice},C_{\rm FGEnt})$; row 2 is the Dice-minus-SDC association with Dice safety; row 3 is the nHD-minus-Dice association with nHD safety; row 4 is the absolute Dice-Exact--M32 association difference under Dice safety. These correlations motivate, but do not prove, the posterior-support mechanism.}",
        r"\label{tab:mechanism-clues}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Diagnostic & " + " & ".join(LABELS[dataset] for dataset in DATASETS) + r" \\",
        r"\midrule",
    ]
    for field, label in rows:
        cells = []
        for dataset in DATASETS:
            cells.append(
                "/".join(
                    f"{values[(dataset, condition)][field]:.3f}"
                    for condition in CONDITIONS
                )
            )
        output.append(label + " & " + " & ".join(cells) + r" \\")
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(output)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.assembled_root)
    values = {
        (dataset, condition): condition_diagnostics(
            _load_condition(root, dataset, condition)
        )
        for dataset in DATASETS
        for condition in CONDITIONS
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(values), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
