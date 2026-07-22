"""Plot risk--coverage curves from explicit binary schema-v2 assemblies.

This is the plotting companion to :mod:`scripts.analyze.main`.  It accepts
only an explicit list of assembled ``records.jsonl`` (or matching manifest)
paths: it never discovers inputs with a glob.  For every dataset/condition it
plots the matched M=32 risk-aligned confidence and SDC under each of Dice,
normalized Hausdorff, and normalized HD95 risk, together with oracle and
random-order references.  ``--all-indexed`` overlays all three M=32 indexed
scores under every risk for the complete cross-risk view.

Example::

    python scripts/plot.py --inputs \
        outputs/binary/pet/clipseg-general/RUN/manifest.json \
        outputs/binary/pet/clipseg-target/RUN/records.jsonl \
        --output-dir figures/binary --png

The curve uses the analytic expectation over a uniformly random order inside
every exact confidence tie.  Therefore the arithmetic mean of its plotted
points is exactly the AURC reported by ``analyze_binary.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze.main import (  # noqa: E402
    EXPECTED_CONDITIONS,
    ConditionData,
    load_condition,
    validate_campaign_bound_conditions,
)
from selectseg.confidence import summarize_aurc  # noqa: E402


ASSEMBLY_ARTIFACT_TYPE = "selectseg.binary_simulation_assembly"
PDF_METADATA = {
    "Title": "Binary selective-segmentation risk-coverage curves",
    "Author": "Anonymous",
    "Subject": "Tie-aware risk-aligned selective risk",
    "Keywords": "selective segmentation, risk coverage, AURC",
    "Creator": "scripts/plot.py",
    "Producer": "Matplotlib",
    # Matplotlib otherwise inserts wall-clock timestamps in every PDF.
    "CreationDate": None,
    "ModDate": None,
}

# Color-blind-safe Okabe--Ito-derived palette.  The two references also differ
# by line style, so the figure remains interpretable in grayscale.
SDC_COLOR = "#009E73"
ORACLE_COLOR = "#202020"
RANDOM_COLOR = "#777777"
GRID_COLOR = "#D8D8D8"
TEXT_COLOR = "#202020"
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
CONDITION_LABELS = {
    "clipseg-general": "CLIPSeg-General",
    "clipseg-target": "CLIPSeg-Target",
    "deeplabv3-target": "DeepLabV3-Target",
    "deeplabv3-external": "DeepLabV3-External",
}
CONDITION_ORDER = {key: index for index, key in enumerate(EXPECTED_CONDITIONS)}
DATASET_ORDER = {
    dataset: index
    for index, dataset in enumerate(
        dict.fromkeys(key[0] for key in EXPECTED_CONDITIONS)
    )
}
COMPLETION_MARKER = "risk_coverage_complete.tex"
RENDER_MANIFEST = "risk_coverage_manifest.json"


@dataclass(frozen=True)
class RiskSpec:
    risk_field: str
    risk_label: str
    score_field: str
    score_label: str
    color: str


RISK_SPECS = (
    RiskSpec(
        "risk_dice",
        "Dice risk",
        "confidence_dice_m32",
        "Dice-M32 (matched)",
        "#0072B2",
    ),
    RiskSpec(
        "risk_nhd",
        "HD risk",
        "confidence_nhd_m32",
        "HD-M32 (matched)",
        "#D55E00",
    ),
    RiskSpec(
        "risk_nhd95",
        "HD95 risk",
        "confidence_nhd95_m32",
        "HD95-M32 (matched)",
        "#CC79A7",
    ),
)
SDC_FIELD = "confidence_sdc"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "explicit assembled records JSONL or matching manifest paths; "
            "directories and automatic discovery are intentionally unsupported"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="figures/binary_risk_coverage",
        help="directory for one deterministic PDF per dataset",
    )
    parser.add_argument(
        "--campaign-lock",
        default=None,
        help="immutable lock required for complete publication figures",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="permit a declared subset for smoke figures",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a 300-dpi PNG beside every publication PDF",
    )
    parser.add_argument(
        "--all-indexed",
        action="store_true",
        help=(
            "overlay Dice-M32, HD-M32, and HD95-M32 under every risk; "
            "the default plots only the matched indexed score in each row"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution when --png is used (default: 300)",
    )
    args = parser.parse_args(argv)
    if isinstance(args.dpi, bool) or args.dpi <= 0:
        parser.error("--dpi must be a positive integer")
    return args


def tie_aware_risk_coverage_curve(confidences, risks):
    """Return the expected discrete curve under random within-tie ordering.

    Coverage is ``k / n`` for ``k=1,...,n``.  If a tie group of size ``g`` has
    mean risk ``mu`` and cumulative risk ``S`` precedes it, the expected
    cumulative risk after accepting ``j`` tied observations is ``S+j*mu``.
    This is the same analytic convention used by
    :func:`selectseg.confidence.tie_aware_expected_aurc`.
    """

    confidence = np.asarray(confidences, dtype=float)
    risk = np.asarray(risks, dtype=float)
    if confidence.ndim != 1 or risk.ndim != 1:
        raise ValueError("confidences and risks must be one-dimensional")
    if confidence.size == 0:
        raise ValueError("confidences and risks must be non-empty")
    if confidence.size != risk.size:
        raise ValueError("confidences and risks must have the same length")
    if not np.isfinite(confidence).all() or not np.isfinite(risk).all():
        raise ValueError("confidences and risks must be finite")

    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_risk = risk[order]
    count = confidence.size

    group_start = np.empty(count, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_confidence[1:] != sorted_confidence[:-1]
    group_index = np.cumsum(group_start) - 1
    group_count = np.bincount(group_index)
    group_sum = np.bincount(group_index, weights=sorted_risk)
    group_mean = group_sum / group_count
    first_position = np.cumsum(group_count) - group_count
    previous_risk = np.cumsum(group_sum) - group_sum

    position = np.arange(count)
    within_group = position - first_position[group_index] + 1
    expected_cumulative = (
        previous_risk[group_index] + within_group * group_mean[group_index]
    )
    coverage = (position + 1) / count
    selective_risk = expected_cumulative / (position + 1)
    return coverage.astype(float), selective_risk.astype(float)


def _jsonl_path_from_input(raw_path) -> Path:
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"explicit plot input does not exist: {path}")
    if path.suffix == ".jsonl":
        return path
    if path.name == "manifest.json":
        return path.parent / "records.jsonl"
    suffix = ".manifest.json"
    if path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)] + ".jsonl")
    raise ValueError(
        "plot inputs must be assembled *.jsonl, manifest.json, or "
        f"*.manifest.json files; got {path}"
    )


def load_assembled_conditions(paths: Iterable[str | Path]):
    """Resolve and strictly validate an explicit collection of assemblies."""

    raw_paths = list(paths)
    if not raw_paths:
        raise ValueError("at least one explicit assembled input is required")
    conditions = []
    for raw_path in raw_paths:
        jsonl_path = _jsonl_path_from_input(raw_path)
        condition = load_condition(jsonl_path)
        artifact_type = condition.manifest.get("artifact_type")
        if artifact_type != ASSEMBLY_ARTIFACT_TYPE:
            raise ValueError(
                f"{condition.manifest_path} is not a final binary assembly: "
                f"artifact_type={artifact_type!r}, expected "
                f"{ASSEMBLY_ARTIFACT_TYPE!r}"
            )
        conditions.append(condition)

    conditions.sort(
        key=lambda item: (item.dataset, item.condition, str(item.jsonl_path))
    )
    identifiers = [(item.dataset, item.condition) for item in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "each dataset/condition pair must be supplied exactly once; "
            f"received {identifiers}"
        )
    return tuple(conditions)


def _curve_and_aurc(confidences, risks):
    coverage, curve = tie_aware_risk_coverage_curve(confidences, risks)
    summary = summarize_aurc(confidences, risks)
    plotted_aurc = float(np.mean(curve))
    if not math.isclose(plotted_aurc, summary.aurc, rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError(
            "risk--coverage curve/AURC convention drift: "
            f"curve mean={plotted_aurc}, analyze_binary={summary.aurc}"
        )
    return coverage, curve, summary


def condition_curves(condition: ConditionData, spec: RiskSpec):
    """Build the four deterministic series plotted in one panel."""

    risks = np.asarray([row[spec.risk_field] for row in condition.rows], dtype=float)
    matched = np.asarray([row[spec.score_field] for row in condition.rows], dtype=float)
    sdc = np.asarray([row[SDC_FIELD] for row in condition.rows], dtype=float)
    coverage, matched_curve, matched_summary = _curve_and_aurc(matched, risks)
    sdc_coverage, sdc_curve, sdc_summary = _curve_and_aurc(sdc, risks)
    oracle_coverage, oracle_curve, oracle_summary = _curve_and_aurc(-risks, risks)
    if not np.array_equal(coverage, sdc_coverage) or not np.array_equal(
        coverage, oracle_coverage
    ):
        raise RuntimeError("risk--coverage grids unexpectedly disagree")
    random_curve = np.full(coverage.shape, float(risks.mean()))
    if not math.isclose(
        float(random_curve.mean()),
        matched_summary.random_aurc,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError("random reference disagrees with analyze_binary")
    return {
        "coverage": coverage,
        "matched": matched_curve,
        "sdc": sdc_curve,
        "oracle": oracle_curve,
        "random": random_curve,
        "matched_aurc": matched_summary.aurc,
        "sdc_aurc": sdc_summary.aurc,
        "oracle_aurc": oracle_summary.aurc,
        "random_aurc": matched_summary.random_aurc,
    }


def condition_all_indexed_curves(condition: ConditionData, spec: RiskSpec):
    """Build all three indexed M32 curves for one evaluation-risk panel."""

    base = condition_curves(condition, spec)
    risks = np.asarray([row[spec.risk_field] for row in condition.rows], dtype=float)
    indexed_curves = {}
    indexed_aurcs = {}
    for indexed_spec in RISK_SPECS:
        if indexed_spec.score_field == spec.score_field:
            curve = base["matched"]
            aurc = base["matched_aurc"]
        else:
            confidence = np.asarray(
                [row[indexed_spec.score_field] for row in condition.rows],
                dtype=float,
            )
            coverage, curve, summary = _curve_and_aurc(confidence, risks)
            if not np.array_equal(coverage, base["coverage"]):
                raise RuntimeError("indexed risk--coverage grids unexpectedly disagree")
            aurc = summary.aurc
        indexed_curves[indexed_spec.score_field] = curve
        indexed_aurcs[indexed_spec.score_field] = aurc
    return {
        "coverage": base["coverage"],
        "indexed": indexed_curves,
        "indexed_aurcs": indexed_aurcs,
        "sdc": base["sdc"],
        "oracle": base["oracle"],
        "random": base["random"],
        "sdc_aurc": base["sdc_aurc"],
        "oracle_aurc": base["oracle_aurc"],
        "random_aurc": base["random_aurc"],
    }


def _pyplot():
    # Headless rendering and a writable cache are required on compute nodes.
    cache = Path(tempfile.gettempdir()) / "selectseg-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "plotting requires Matplotlib; install the project with .[plots]"
        ) from error
    return plt


def _dataset_slug(dataset: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")
    if not slug:
        raise ValueError(
            f"dataset name cannot form a safe output filename: {dataset!r}"
        )
    return slug


def _draw_panel(axis, condition, spec, *, all_indexed=False):
    curves = (
        condition_all_indexed_curves(condition, spec)
        if all_indexed
        else condition_curves(condition, spec)
    )
    coverage = curves["coverage"]
    if all_indexed:
        for indexed_spec in RISK_SPECS:
            axis.plot(
                coverage,
                curves["indexed"][indexed_spec.score_field],
                color=indexed_spec.color,
                linewidth=2.0,
                label=indexed_spec.score_label.replace(" (matched)", ""),
                zorder=4,
            )
    else:
        axis.plot(
            coverage,
            curves["matched"],
            color=spec.color,
            linewidth=2.0,
            label=spec.score_label,
            zorder=4,
        )
    axis.plot(
        coverage,
        curves["sdc"],
        color=SDC_COLOR,
        linewidth=1.8,
        linestyle="-.",
        label="SDC",
        zorder=3,
    )
    axis.plot(
        coverage,
        curves["oracle"],
        color=ORACLE_COLOR,
        linewidth=1.5,
        linestyle="--",
        label="Oracle",
        zorder=2,
    )
    axis.plot(
        coverage,
        curves["random"],
        color=RANDOM_COLOR,
        linewidth=1.4,
        linestyle=":",
        label="Random",
        zorder=1,
    )
    if all_indexed:
        short_labels = {
            "confidence_dice_m32": "D",
            "confidence_nhd_m32": "H",
            "confidence_nhd95_m32": "H95",
        }
        values = {
            short_labels[item.score_field]: (
                100 * curves["indexed_aurcs"][item.score_field]
            )
            for item in RISK_SPECS
        }
        annotation = (
            "AURC ×100\n"
            f"D {values['D']:.2f} | H {values['H']:.2f}\n"
            f"H95 {values['H95']:.2f} | SDC {100 * curves['sdc_aurc']:.2f}"
        )
    else:
        annotation = (
            "AURC ×100\n"
            f"{spec.score_label.split()[0]} "
            f"{100 * curves['matched_aurc']:.3f}"
            f"  |  SDC {100 * curves['sdc_aurc']:.3f}"
        )
    axis.text(
        0.98,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1))
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1))
    axis.grid(color=GRID_COLOR, linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=7.5, colors=TEXT_COLOR)
    return curves


def render_dataset(
    dataset,
    conditions,
    output_dir,
    *,
    png=False,
    dpi=300,
    all_indexed=False,
    render_spec_sha256=None,
):
    """Render one dataset's three-risk by condition grid."""

    conditions = tuple(
        sorted(
            conditions,
            key=lambda item: CONDITION_ORDER.get(
                (item.dataset, item.condition), len(CONDITION_ORDER)
            ),
        )
    )
    if not conditions:
        raise ValueError("cannot render a dataset with no conditions")
    if any(condition.dataset != dataset for condition in conditions):
        raise ValueError("all rendered conditions must belong to the named dataset")

    plt = _pyplot()
    column_count = len(conditions)
    width = max(4.8, 3.15 * column_count)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            len(RISK_SPECS),
            column_count,
            figsize=(width, 7.25),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for column, condition in enumerate(conditions):
            axes[0, column].set_title(
                CONDITION_LABELS.get(
                    condition.condition, condition.condition.replace("-", " ")
                ),
                fontsize=9.5,
                pad=7,
            )
            for row, spec in enumerate(RISK_SPECS):
                _draw_panel(
                    axes[row, column],
                    condition,
                    spec,
                    all_indexed=all_indexed,
                )
                if column == 0:
                    axes[row, column].set_ylabel(spec.risk_label, fontsize=8.5)
                if row == len(RISK_SPECS) - 1:
                    axes[row, column].set_xlabel("Coverage", fontsize=8.5)

        handles_by_label = {}
        for row in range(len(RISK_SPECS)):
            handles, labels = axes[row, 0].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                handles_by_label.setdefault(label, handle)
        legend_labels = [
            (
                spec.score_label.replace(" (matched)", "")
                if all_indexed
                else spec.score_label
            )
            for spec in RISK_SPECS
        ]
        legend_labels.extend(["SDC", "Oracle", "Random"])
        legend_handles = [handles_by_label[label] for label in legend_labels]
        figure.suptitle(
            f"Selective risk-coverage: {DATASET_LABELS.get(dataset, dataset)}",
            fontsize=12,
            fontweight="bold",
        )
        legend_columns = 6 if column_count >= 4 else 3
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.952),
            ncol=legend_columns,
            frameon=False,
            fontsize=8,
            handlelength=2.6,
            columnspacing=1.3,
        )
        figure.subplots_adjust(
            left=0.09 if column_count > 1 else 0.18,
            right=0.985,
            bottom=0.075,
            top=(0.81 if column_count == 1 else (0.875 if column_count >= 4 else 0.82)),
            wspace=0.12,
            hspace=0.17,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = "all_indexed_" if all_indexed else ""
        stem = f"risk_coverage_{mode}{_dataset_slug(dataset)}"
        pdf_path = output_dir / f"{stem}.pdf"
        pdf_metadata = dict(PDF_METADATA)
        if render_spec_sha256 is not None:
            pdf_metadata["Subject"] = (
                f"Source render-spec SHA-256 {render_spec_sha256}"
            )
        figure.savefig(pdf_path, format="pdf", metadata=pdf_metadata)
        paths = [pdf_path]
        if png:
            png_path = output_dir / f"{stem}.png"
            figure.savefig(
                png_path,
                format="png",
                dpi=dpi,
                metadata={"Software": "scripts/plot.py"},
            )
            paths.append(png_path)
        plt.close(figure)
    return tuple(paths)


def render_conditions(
    conditions,
    output_dir,
    *,
    png=False,
    dpi=300,
    all_indexed=False,
    render_spec_sha256=None,
):
    """Render deterministic dataset figures from validated conditions."""

    conditions = tuple(conditions)
    if not conditions:
        raise ValueError("at least one assembled condition is required")
    by_dataset = {}
    for condition in conditions:
        by_dataset.setdefault(condition.dataset, []).append(condition)
    slugs = [_dataset_slug(dataset) for dataset in by_dataset]
    if len(slugs) != len(set(slugs)):
        raise ValueError("dataset names collide after output-filename normalization")

    outputs = []
    for dataset in sorted(
        by_dataset, key=lambda name: (DATASET_ORDER.get(name, len(DATASET_ORDER)), name)
    ):
        outputs.extend(
            render_dataset(
                dataset,
                by_dataset[dataset],
                output_dir,
                png=png,
                dpi=dpi,
                all_indexed=all_indexed,
                render_spec_sha256=render_spec_sha256,
            )
        )
    return tuple(outputs)


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_sha256(value):
    """Hash one JSON-compatible value deterministically."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def provenance_sha256(provenance):
    """Hash the validated, portable source-artifact bundle deterministically."""

    return canonical_sha256(provenance)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publication_render_spec(provenance):
    source_bundle_sha = provenance_sha256(provenance)
    outputs = [
        f"risk_coverage_all_indexed_{dataset}.pdf" for dataset in DATASET_ORDER
    ]
    return {
        "schema_version": 1,
        "artifact_type": "selectseg.risk_coverage_render_spec",
        "campaign_lock_sha256": provenance["campaign_lock"]["sha256"],
        "source_artifact_bundle_sha256": source_bundle_sha,
        "source_artifact_bundle": provenance,
        "plot_source_sha256": _sha256_file(__file__),
        "parameters": {
            "all_indexed": True,
            "png": False,
            "risk_fields": [spec.risk_field for spec in RISK_SPECS],
            "score_fields": [spec.score_field for spec in RISK_SPECS],
            "dataset_order": list(DATASET_ORDER),
        },
        "outputs": outputs,
    }


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    conditions = load_assembled_conditions(args.inputs)
    identifiers = {(item.dataset, item.condition) for item in conditions}
    if not args.allow_incomplete:
        if args.campaign_lock is None:
            raise ValueError("complete figures require --campaign-lock")
        if identifiers != set(EXPECTED_CONDITIONS):
            raise ValueError("complete figures require exactly 16 declared conditions")
    provenance = None
    if args.campaign_lock is not None:
        provenance = validate_campaign_bound_conditions(conditions, args.campaign_lock)
    output_dir = Path(args.output_dir)
    marker = output_dir / COMPLETION_MARKER
    manifest_path = output_dir / RENDER_MANIFEST
    render_spec = None
    render_spec_sha = None
    if not args.allow_incomplete:
        if not args.all_indexed or args.png:
            raise ValueError(
                "complete manuscript figures require --all-indexed without --png"
            )
        render_spec = publication_render_spec(provenance)
        render_spec_sha = canonical_sha256(render_spec)
        marker.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    outputs = render_conditions(
        conditions,
        output_dir,
        png=args.png,
        dpi=args.dpi,
        all_indexed=args.all_indexed,
        render_spec_sha256=render_spec_sha,
    )
    for path in outputs:
        print(f"saved {path}")
    if not args.allow_incomplete:
        expected = render_spec["outputs"]
        if [path.name for path in outputs] != expected:
            raise RuntimeError("publication figure set is incomplete")
        lock_sha = provenance["campaign_lock"]["sha256"]
        source_bundle_sha = provenance_sha256(provenance)
        render_manifest = {
            "schema_version": 1,
            "artifact_type": "selectseg.risk_coverage_render_manifest",
            "render_spec_sha256": render_spec_sha,
            "render_spec": render_spec,
            "outputs": [
                {"path": path.name, "sha256": _sha256_file(path)} for path in outputs
            ],
        }
        _atomic_write_text(
            manifest_path,
            json.dumps(
                render_manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )
        render_manifest_sha = _sha256_file(manifest_path)
        _atomic_write_text(
            marker,
            "% AUTO-GENERATED completion sentinel; DO NOT EDIT.\n"
            f"% Campaign lock SHA-256: {lock_sha}\n"
            f"% Source artifact bundle SHA-256: {source_bundle_sha}\n"
            f"% Render spec SHA-256: {render_spec_sha}\n"
            f"% Render manifest SHA-256: {render_manifest_sha}\n"
            rf"\def\RiskCoverageSourceBundleSHA{{{source_bundle_sha}}}" + "\n"
            rf"\def\RiskCoverageRenderSpecSHA{{{render_spec_sha}}}" + "\n"
            rf"\def\RiskCoverageRenderManifestSHA{{{render_manifest_sha}}}" + "\n"
            rf"\def\RiskCoverageCampaignSHA{{{lock_sha}}}" + "\n",
        )
        print(f"saved {manifest_path}")
        print(f"saved {marker}")
    return outputs


if __name__ == "__main__":
    main()
