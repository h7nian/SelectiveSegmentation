"""Render mechanically selected qualitative cases from frozen binary artifacts.

Run this only after :mod:`scripts.select_cases` has published
an immutable ``selection.json``.  The renderer verifies that selection's
content ID, opens only the selected lock-bound NPZ payloads, and creates one
lossless four-column PNG per dataset plus a compact TeX include.  Raw source
images are intentionally not rediscovered from mutable dataset directories.

Outputs are content addressed and existing render directories are never
replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import textwrap
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from scripts.select_cases import (
    ARTIFACT_TYPE as SELECTION_ARTIFACT_TYPE,
    CASE_ORDER,
    DATASET_ORDER,
    SCHEMA_VERSION as SELECTION_SCHEMA_VERSION,
    validate_selection_id,
)
from selectseg.artifacts import (
    PROBABILITY_KEY,
    TRUTH_KEY,
    load_binary_artifact,
    sha256_file,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_qualitative_case_render"
REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_SIZE = 270
LEFT_MARGIN = 280
HEADER_HEIGHT = 52
ROW_HEIGHT = 326
GAP = 14
CASE_LABELS = {
    "dice_vs_nhd_rank_disagreement": "Dice-M32 vs nHD-M32\nrank disagreement",
    "nhd_vs_nhd95_rank_disagreement": "nHD-M32 vs nHD95-M32\nrank disagreement",
    "empty_action": "Deployed empty action",
    "confident_failure": "Confident failure",
}
COLUMN_LABELS = (
    "Foreground probability",
    "Reference truth",
    "Deployed action (gamma=0.5)",
    "Boundaries on probability",
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--output-root",
        help="default: a rendered/ directory beside selection.json",
    )
    parser.add_argument(
        "--render-manifest",
        help="publish an existing verified render package instead of rendering again",
    )
    parser.add_argument(
        "--paper-output-dir",
        help="optionally publish stable manuscript TeX/PNG names into this directory",
    )
    return parser.parse_args(argv)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _regular_source_without_symlink_ancestors(
    path: str | os.PathLike[str], *, expected_name: str
) -> Path:
    """Return one absolute regular file while preserving its lexical trust path."""

    source = Path(os.path.abspath(path))
    if (
        source.name != expected_name
        or any(candidate.is_symlink() for candidate in (source, *source.parents))
        or not source.is_file()
    ):
        raise FileNotFoundError(
            f"expected a regular {expected_name} with no symlink ancestors: {source}"
        )
    return source


def _load_selection(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    source = _regular_source_without_symlink_ancestors(
        path, expected_name="selection.json"
    )
    report = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(report, dict):
        raise ValueError("selection.json must contain one JSON object")
    if report.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("selection schema version is unsupported")
    if report.get("artifact_type") != SELECTION_ARTIFACT_TYPE:
        raise ValueError("selection artifact_type is unsupported")
    validate_selection_id(report)
    counts = report.get("condition_counts")
    if counts != {
        "validated_conditions": 16,
        "eligible_target_conditions": 10,
        "datasets": 5,
    }:
        raise ValueError("selection does not bind the canonical 16/10/5 scope")
    datasets = report.get("datasets")
    if not isinstance(datasets, list) or [
        item.get("dataset") for item in datasets
    ] != list(DATASET_ORDER):
        raise ValueError("selection datasets are absent or out of canonical order")
    for dataset in datasets:
        cases = dataset.get("cases")
        if not isinstance(cases, list) or [
            case.get("case_type") for case in cases
        ] != list(CASE_ORDER):
            raise ValueError("selection cases are absent or out of declared order")
        for case in cases:
            if case.get("status") not in {"selected", "unavailable"}:
                raise ValueError("case status must be selected or unavailable")
            if (
                case["status"] == "selected"
                and case.get("dataset") != dataset["dataset"]
            ):
                raise ValueError("selected case dataset differs from its group")
    return source, report


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/select_cases.py",
        REPO_ROOT / "selectseg/artifacts.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _safe_payload_path(artifact_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".npz":
        raise ValueError(f"unsafe selected payload path {relative!r}")
    root = Path(os.path.abspath(artifact_dir))
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink():
            raise ValueError(f"selected artifact root traverses a symlink: {ancestor}")
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"selected payload path escapes the artifact directory: {relative!r}"
        ) from error
    cursor = root
    for part in pure.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"selected payload path traverses a symlink: {cursor}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"selected payload is not a regular file: {path}")
    return path


def load_selected_arrays(
    case: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one selected payload after validating every bound identity and hash."""

    provenance = case.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selected case lacks provenance")
    manifest_path = _regular_source_without_symlink_ancestors(
        _resolve_repo_path(str(provenance["artifact_manifest_path"])),
        expected_name="manifest.json",
    )
    artifact = load_binary_artifact(manifest_path, validate_payloads=False)
    if artifact.manifest_sha256 != provenance.get("artifact_manifest_sha256"):
        raise ValueError("artifact manifest hash differs from selection")
    if artifact.manifest.get("artifact_id") != provenance.get("artifact_id"):
        raise ValueError("artifact ID differs from selection")
    for field in ("dataset", "condition"):
        if artifact.manifest.get(field) != case.get(field):
            raise ValueError(f"artifact {field} differs from selected case")

    entries = artifact.manifest["samples"]
    matches = [entry for entry in entries if entry["sample_id"] == case["sample_id"]]
    if len(matches) != 1:
        raise ValueError("selected sample_id is not unique in frozen artifact")
    entry = matches[0]
    if entry["index"] != case["image_index"]:
        raise ValueError("selected image_index differs from artifact manifest")
    if entry["path"] != provenance.get("sample_payload_path"):
        raise ValueError("selected payload path differs from artifact manifest")
    if entry["sha256"] != provenance.get("sample_payload_sha256"):
        raise ValueError("selected payload hash differs from artifact manifest")
    payload_path = _safe_payload_path(artifact.artifact_dir, entry["path"])
    actual_hash = sha256_file(payload_path)
    if actual_hash != entry["sha256"]:
        raise ValueError("selected NPZ payload hash mismatch")

    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != {PROBABILITY_KEY, TRUTH_KEY}:
            raise ValueError("selected NPZ has an unsupported member schema")
        probability = np.asarray(payload[PROBABILITY_KEY])
        truth = np.asarray(payload[TRUTH_KEY])
    expected_shape = (int(case["height"]), int(case["width"]))
    if probability.shape != expected_shape or truth.shape != expected_shape:
        raise ValueError("selected arrays differ from recorded native shape")
    if probability.dtype != np.float32 or truth.dtype != np.uint8:
        raise ValueError("selected arrays differ from frozen artifact dtypes")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("selected probability values must be finite and in [0, 1]")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("selected truth must be binary")
    return (
        probability,
        truth.astype(bool),
        {
            "dataset": case["dataset"],
            "condition": case["condition"],
            "sample_id": case["sample_id"],
            "artifact_id": provenance["artifact_id"],
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "payload_path": provenance["sample_payload_path"],
            "payload_sha256": actual_hash,
        },
    )


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - compatibility with old Pillow
        return ImageFont.load_default()


def _fit(image: Image.Image, size: int, *, nearest: bool) -> Image.Image:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR,
    )
    canvas = Image.new(image.mode, (size, size), 0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _probability_tile(probability: np.ndarray) -> Image.Image:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _fit(Image.fromarray(values, mode="L"), CELL_SIZE, nearest=False).convert(
        "RGB"
    )


def _mask_tile(mask: np.ndarray) -> Image.Image:
    values = mask.astype(np.uint8) * 255
    return _fit(Image.fromarray(values, mode="L"), CELL_SIZE, nearest=True).convert(
        "RGB"
    )


def _boundary(mask: Image.Image) -> np.ndarray:
    foreground = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    eroded = np.asarray(mask.convert("L").filter(ImageFilter.MinFilter(3))) > 127
    edge = foreground & ~eroded
    thick = Image.fromarray(edge.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(3)
    )
    return np.asarray(thick) > 127


def _boundary_tile(
    probability: np.ndarray, truth: np.ndarray, action: np.ndarray
) -> Image.Image:
    base = _probability_tile(probability)
    truth_resized = _fit(
        Image.fromarray(truth.astype(np.uint8) * 255, mode="L"),
        CELL_SIZE,
        nearest=True,
    )
    action_resized = _fit(
        Image.fromarray(action.astype(np.uint8) * 255, mode="L"),
        CELL_SIZE,
        nearest=True,
    )
    truth_edge = _boundary(truth_resized)
    action_edge = _boundary(action_resized)
    pixels = np.asarray(base).copy()
    pixels[truth_edge] = (0, 220, 80)
    pixels[action_edge] = (235, 30, 190)
    pixels[truth_edge & action_edge] = (255, 215, 0)
    return Image.fromarray(pixels, mode="RGB")


def _objective_label(case: Mapping[str, Any]) -> str:
    value = float(case["selection_objective"])
    if case["case_type"].endswith("rank_disagreement"):
        return f"normalized rank gap={value:.3f}"
    if case["case_type"] == "empty_action":
        loss = case["selection_objective_details"]["matched_loss"]
        return f"largest matched loss={value:.3f} ({loss})"
    loss = case["selection_objective_details"]["matched_loss"]
    return f"observed-predicted gap={value:.3f} ({loss})"


def compose_dataset_panel(
    dataset: str,
    cases: Sequence[Mapping[str, Any]],
    arrays: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> Image.Image:
    """Compose a deterministic panel from already validated selected arrays."""

    if not cases or not any(case["status"] == "selected" for case in cases):
        raise ValueError(f"dataset {dataset} has no renderable selected cases")
    width = LEFT_MARGIN + 4 * CELL_SIZE + 5 * GAP
    height = HEADER_HEIGHT + len(cases) * ROW_HEIGHT + GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(23)
    label_font = _font(18)
    small_font = _font(15)
    for column, title in enumerate(COLUMN_LABELS):
        x = LEFT_MARGIN + GAP + column * (CELL_SIZE + GAP)
        draw.text((x + 4, 12), title, fill="black", font=label_font)

    for row_index, case in enumerate(cases):
        y = HEADER_HEIGHT + row_index * ROW_HEIGHT
        case_label = CASE_LABELS[str(case["case_type"])]
        draw.text(
            (12, y + 10),
            case_label,
            fill="black",
            font=title_font,
        )
        if case["status"] == "unavailable":
            draw.multiline_text(
                (12, y + 76),
                "Unavailable by rule:\n" + textwrap.fill(str(case["reason"]), 30),
                fill=(60, 60, 60),
                font=small_font,
                spacing=6,
            )
            for column in range(4):
                x = LEFT_MARGIN + GAP + column * (CELL_SIZE + GAP)
                tile = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (242, 242, 242))
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text(
                    (CELL_SIZE // 2 - 48, CELL_SIZE // 2 - 8),
                    "not available",
                    fill=(100, 100, 100),
                    font=small_font,
                )
                canvas.paste(tile, (x, y + GAP))
                draw.rectangle(
                    (x, y + GAP, x + CELL_SIZE - 1, y + GAP + CELL_SIZE - 1),
                    outline=(170, 170, 170),
                    width=1,
                )
            if row_index < len(cases) - 1:
                draw.line(
                    (8, y + ROW_HEIGHT - 2, width - 8, y + ROW_HEIGHT - 2),
                    fill=(210, 210, 210),
                )
            continue

        key = str(case["condition"]), str(case["sample_id"])
        probability, truth = arrays[key]
        action = probability >= 0.5
        tiles = (
            _probability_tile(probability),
            _mask_tile(truth),
            _mask_tile(action),
            _boundary_tile(probability, truth, action),
        )
        sample_id = str(case["sample_id"])
        if len(sample_id) > 22:
            sample_id = sample_id[:19] + "..."
        description = (
            f"{case['condition']}\n"
            f"sample: {sample_id}\n"
            f"{_objective_label(case)}\n"
            f"risk D/H/H95: {case['risks']['risk_dice']:.3f} / "
            f"{case['risks']['risk_nhd']:.3f} / "
            f"{case['risks']['risk_nhd95']:.3f}"
        )
        draw.multiline_text(
            (12, y + (68 if "\n" in case_label else 48)),
            description,
            fill=(35, 35, 35),
            font=small_font,
            spacing=6,
        )
        for column, tile in enumerate(tiles):
            x = LEFT_MARGIN + GAP + column * (CELL_SIZE + GAP)
            canvas.paste(tile, (x, y + GAP))
            draw.rectangle(
                (x, y + GAP, x + CELL_SIZE - 1, y + GAP + CELL_SIZE - 1),
                outline=(90, 90, 90),
                width=1,
            )
        legend_x = LEFT_MARGIN + GAP + 3 * (CELL_SIZE + GAP) + 6
        legend_y = y + CELL_SIZE + 20
        draw.line(
            (legend_x, legend_y, legend_x + 25, legend_y), fill=(0, 220, 80), width=4
        )
        draw.text((legend_x + 32, legend_y - 8), "truth", fill="black", font=small_font)
        draw.line(
            (legend_x + 95, legend_y, legend_x + 120, legend_y),
            fill=(235, 30, 190),
            width=4,
        )
        draw.text(
            (legend_x + 127, legend_y - 8), "action", fill="black", font=small_font
        )
        if row_index < len(cases) - 1:
            draw.line(
                (8, y + ROW_HEIGHT - 2, width - 8, y + ROW_HEIGHT - 2),
                fill=(210, 210, 210),
            )
    return canvas


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _tex_root(destination: Path) -> str:
    try:
        relative = destination.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return destination.resolve().as_posix()
    return f"../{relative}"


def _sha256_hex(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return normalized


def _selection_campaign_sha(selection: Mapping[str, Any]) -> str:
    provenance = selection.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selection.provenance must be an object")
    campaign_lock = provenance.get("campaign_lock")
    if not isinstance(campaign_lock, Mapping):
        raise ValueError("selection.provenance.campaign_lock must be an object")
    return _sha256_hex(
        campaign_lock.get("sha256"),
        location="selection.provenance.campaign_lock.sha256",
    )


def _render_id(selection_sha256: str, renderer_source_sha256: str) -> str:
    """Derive the content address shared by renderer and publisher."""

    selection_sha256 = _sha256_hex(selection_sha256, location="selection_sha256")
    renderer_source_sha256 = _sha256_hex(
        renderer_source_sha256, location="renderer_source_sha256"
    )
    return hashlib.sha256(
        (selection_sha256 + "\0" + renderer_source_sha256).encode("ascii")
    ).hexdigest()[:16]


def _render_tex(
    destination: Path,
    selection_id: str,
    *,
    selection_sha256: str,
    campaign_lock_sha256: str,
) -> str:
    root = _tex_root(destination)
    pieces = [
        "% Auto-generated qualitative artifact; do not edit by hand.",
        f"% Selection JSON SHA-256: {selection_sha256}",
        f"% Campaign lock SHA-256: {campaign_lock_sha256}",
        f"% selection_id={selection_id}; render_id={destination.name}",
        rf"\providecommand{{\QualitativeArtifactRoot}}{{{root}}}",
    ]
    for dataset in DATASET_ORDER:
        display = {
            "pet": "Oxford Pet",
            "kvasir": "Kvasir-SEG",
            "fives": "FIVES",
            "isic": "ISIC 2018",
            "tn3k": "TN3K",
        }[dataset]
        pieces.extend(
            [
                r"\begin{figure*}[t]",
                r"\centering",
                rf"\includegraphics[width=\textwidth]{{\QualitativeArtifactRoot/{dataset}.png}}",
                (
                    r"\caption{Mechanically selected diagnostics on "
                    + _tex_escape(display)
                    + r". Rows follow predeclared numerical rules rather than visual "
                    r"appeal and are not representative examples. Frozen artifacts "
                    r"bind only probability and truth arrays, so the panels show the "
                    r"probability map, reference truth, deployed $\gamma=0.5$ action, "
                    r"and boundaries over the probability map (truth: green; action: "
                    r"magenta; overlap: yellow).}"
                ),
                rf"\label{{fig:qualitative-{dataset}}}",
                r"\end{figure*}",
            ]
        )
    return "\n".join(pieces) + "\n"


def _render_manuscript_tex(
    selection: Mapping[str, Any],
    *,
    selection_sha256: str,
    render_id: str,
    renderer_source_sha256: str,
    source_render_manifest_sha256: str,
) -> str:
    selection_id = selection["selection_id"]
    campaign_lock_sha256 = _selection_campaign_sha(selection)
    empty_action_datasets = sum(
        any(
            case.get("case_type") == "empty_action" and case.get("status") == "selected"
            for case in dataset["cases"]
        )
        for dataset in selection["datasets"]
    )
    if empty_action_datasets:
        empty_sentence = (
            f" The fixed rule finds an empty deployed action in {empty_action_datasets} "
            f"of {len(DATASET_ORDER)} dataset panels."
        )
    else:
        empty_sentence = (
            " No target condition has an empty deployed action, which remains "
            "explicitly unavailable."
        )
    display_names = {
        "pet": "Oxford-IIIT Pet",
        "kvasir": "Kvasir-SEG",
        "fives": "FIVES",
        "isic": "ISIC 2018",
        "tn3k": "TN3K",
    }
    pieces = [
        "% AUTO-GENERATED manuscript qualitative figures; DO NOT EDIT.",
        f"% Selection JSON SHA-256: {selection_sha256}",
        f"% Campaign lock SHA-256: {campaign_lock_sha256}",
        f"% Renderer source SHA-256: {renderer_source_sha256}",
        f"% Source render manifest SHA-256: {source_render_manifest_sha256}",
        f"% selection_id={selection_id}; render_id={render_id}",
        "% Cases are selected mechanically; do not replace them by visual inspection.",
    ]
    for index, dataset in enumerate(DATASET_ORDER):
        pieces.extend(
            [
                r"\begin{figure*}[p]",
                r"  \centering",
                rf"  \includegraphics[width=\textwidth]{{Figures/qualitative_{dataset}.png}}",
            ]
        )
        if index == 0:
            caption = (
                f"Mechanically selected {display_names[dataset]} diagnostics. Rows follow "
                "stated post-analysis numerical rules rather than visual appeal and are not "
                "representative examples. Panels show the probability map, reference truth, "
                "deployed \\(\\gamma=0.5\\) action, and boundaries over the probability map "
                "(truth: green; action: magenta; overlap: yellow)." + empty_sentence
            )
        else:
            caption = (
                f"Mechanically selected {display_names[dataset]} diagnostics under the same "
                "fixed rules and display convention as Figure~\\ref{fig:qualitative-pet}."
            )
        pieces.extend(
            [
                rf"  \caption{{{caption}}}",
                rf"  \label{{fig:qualitative-{dataset}}}",
                r"\end{figure*}",
            ]
        )
    return "\n".join(pieces) + "\n"


def render_package(
    selection_path: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    selection_file, selection = _load_selection(selection_path)
    selection_sha = sha256_file(selection_file)
    campaign_lock_sha = _selection_campaign_sha(selection)
    renderer_sha = _source_sha256()
    render_id = _render_id(selection_sha, renderer_sha)
    root = (
        Path(output_root)
        if output_root is not None
        else selection_file.parent / "rendered"
    )
    if root.is_symlink():
        raise ValueError(f"render output root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / render_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite render package: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{render_id}.tmp-", dir=root))
    payload_provenance: dict[tuple[str, str, str], dict] = {}
    try:
        image_records = []
        for dataset_entry in selection["datasets"]:
            arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
            for case in dataset_entry["cases"]:
                if case["status"] != "selected":
                    continue
                key = (case["dataset"], case["condition"], case["sample_id"])
                if key not in payload_provenance:
                    probability, truth, provenance = load_selected_arrays(case)
                    payload_provenance[key] = provenance
                    arrays[(case["condition"], case["sample_id"])] = (
                        probability,
                        truth,
                    )
                else:
                    probability, truth, _ = load_selected_arrays(case)
                    arrays[(case["condition"], case["sample_id"])] = (
                        probability,
                        truth,
                    )
            panel = compose_dataset_panel(
                dataset_entry["dataset"], dataset_entry["cases"], arrays
            )
            image_path = temporary / f"{dataset_entry['dataset']}.png"
            panel.save(image_path, format="PNG", optimize=False, compress_level=9)
            image_records.append(
                {
                    "dataset": dataset_entry["dataset"],
                    "path": image_path.name,
                    "sha256": sha256_file(image_path),
                    "width": panel.width,
                    "height": panel.height,
                }
            )

        tex_path = temporary / "qualitative_cases.tex"
        tex_path.write_text(
            _render_tex(
                destination,
                selection["selection_id"],
                selection_sha256=selection_sha,
                campaign_lock_sha256=campaign_lock_sha,
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "render_id": render_id,
            "selection_id": selection["selection_id"],
            "selection_path": str(selection_file),
            "selection_sha256": selection_sha,
            "renderer_source_sha256": renderer_sha,
            "source_rgb_used": False,
            "source_rgb_reason": (
                "the immutable frozen artifacts bind probability and truth arrays "
                "but do not bind raw RGB payload paths or hashes"
            ),
            "interpretation": (
                "predeclared diagnostic selections, not representative examples"
            ),
            "selected_payloads": sorted(
                payload_provenance.values(),
                key=lambda item: (
                    item["dataset"],
                    item["condition"],
                    item["sample_id"],
                ),
            ),
            "images": image_records,
            "tex": {
                "path": tex_path.name,
                "sha256": sha256_file(tex_path),
            },
        }
        manifest_path = temporary / "render_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination / "render_manifest.json"


def _load_render_manifest(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    source = _regular_source_without_symlink_ancestors(
        path, expected_name="render_manifest.json"
    )
    manifest = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(manifest, dict):
        raise ValueError("render_manifest.json must contain one JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("render manifest schema version is unsupported")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("render manifest artifact_type is unsupported")
    selection_sha = _sha256_hex(
        manifest.get("selection_sha256"),
        location="render_manifest.selection_sha256",
    )
    renderer_source_sha = _sha256_hex(
        manifest.get("renderer_source_sha256"),
        location="render_manifest.renderer_source_sha256",
    )
    render_id = manifest.get("render_id")
    expected_render_id = _render_id(selection_sha, renderer_source_sha)
    if render_id != expected_render_id or render_id != source.parent.name:
        raise ValueError(
            "render manifest ID differs from its selection/source content address"
        )
    return source, manifest


def publish_manuscript_package(
    render_manifest_path: str | os.PathLike[str],
    selection_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Publish verified content-addressed panels under stable manuscript names."""

    manifest_path, manifest = _load_render_manifest(render_manifest_path)
    selection_file, selection = _load_selection(selection_path)
    selection_sha = sha256_file(selection_file)
    source_render_manifest_sha = sha256_file(manifest_path)
    renderer_source_sha = _sha256_hex(
        manifest.get("renderer_source_sha256"),
        location="render_manifest.renderer_source_sha256",
    )
    if manifest.get("selection_id") != selection["selection_id"]:
        raise ValueError("render manifest selection_id differs from selection.json")
    if manifest.get("selection_sha256") != selection_sha:
        raise ValueError(
            "render manifest selection SHA-256 differs from selection.json"
        )
    expected_render_id = _render_id(selection_sha, renderer_source_sha)
    if manifest.get("render_id") != expected_render_id:
        raise ValueError(
            "render manifest ID differs from its selection/source content address"
        )

    images = manifest.get("images")
    if not isinstance(images, list) or [item.get("dataset") for item in images] != list(
        DATASET_ORDER
    ):
        raise ValueError("render manifest images are absent or out of canonical order")

    destination = Path(output_dir)
    if destination.is_symlink():
        raise ValueError(f"paper output directory must not be a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".qualitative-paper.tmp-", dir=destination)
    )
    staged: list[tuple[Path, Path]] = []
    try:
        for image_record in images:
            dataset = image_record["dataset"]
            expected_name = f"{dataset}.png"
            if image_record.get("path") != expected_name:
                raise ValueError(f"render image path for {dataset} is not canonical")
            expected_sha = _sha256_hex(
                image_record.get("sha256"),
                location=f"render_manifest.images[{dataset}].sha256",
            )
            source = manifest_path.parent / expected_name
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"render image is not a regular file: {source}")
            if sha256_file(source) != expected_sha:
                raise ValueError(f"render image SHA-256 mismatch: {source}")
            staged_path = temporary / f"qualitative_{dataset}.png"
            shutil.copyfile(source, staged_path)
            staged.append((staged_path, destination / staged_path.name))

        tex_path = temporary / "qualitative_cases.tex"
        tex_path.write_text(
            _render_manuscript_tex(
                selection,
                selection_sha256=selection_sha,
                render_id=manifest["render_id"],
                renderer_source_sha256=renderer_source_sha,
                source_render_manifest_sha256=source_render_manifest_sha,
            ),
            encoding="utf-8",
        )
        staged.append((tex_path, destination / tex_path.name))
        public_outputs = [
            {"path": target.name, "sha256": sha256_file(source)}
            for source, target in staged
        ]
        public_manifest_path = temporary / "qualitative_manifest.json"
        public_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "selectseg.binary_qualitative_manuscript",
                    "selection_id": selection["selection_id"],
                    "selection_sha256": selection_sha,
                    "campaign_lock_sha256": _selection_campaign_sha(selection),
                    "render_id": manifest["render_id"],
                    "renderer_source_sha256": renderer_source_sha,
                    "source_render_manifest_sha256": source_render_manifest_sha,
                    "outputs": public_outputs,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        staged.append(
            (
                public_manifest_path,
                destination / "qualitative_manifest.json",
            )
        )
        for _, target in staged:
            if target.is_symlink():
                raise ValueError(
                    f"refusing to replace symlinked manuscript output: {target}"
                )
            if target.exists() and not target.is_file():
                raise ValueError(
                    f"refusing to replace non-file manuscript output: {target}"
                )

        # ``qualitative_cases.tex`` is the manuscript's visibility guard, and
        # the public manifest is the mirror validator's guard.  All source
        # validation and staging above happens while the old bundle is intact;
        # immediately before publishing any replacement, remove both guards so
        # an interruption can only leave a fail-closed manuscript package.
        tex_target = destination / "qualitative_cases.tex"
        manifest_target = destination / "qualitative_manifest.json"
        for guard in (tex_target, manifest_target):
            if guard.exists():
                guard.unlink()

        # Publish every payload and its hash manifest first, then expose the
        # TeX guard in one final atomic rename.  The stable-name bundle cannot
        # be swapped as one directory, but it is never manuscript-visible in a
        # partially replaced state.
        publication_order = [
            item for item in staged if item[1].name != "qualitative_cases.tex"
        ]
        publication_order.append(
            next(item for item in staged if item[1].name == "qualitative_cases.tex")
        )
        for source, target in publication_order:
            os.replace(source, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return tuple(target for _, target in staged)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.render_manifest is not None:
        if args.output_root is not None:
            raise ValueError("--output-root cannot be used with --render-manifest")
        if args.paper_output_dir is None:
            raise ValueError("--render-manifest requires --paper-output-dir")
        outputs = publish_manuscript_package(
            args.render_manifest,
            args.selection,
            args.paper_output_dir,
        )
        for output in outputs:
            print(output)
        return

    output = render_package(args.selection, output_root=args.output_root)
    if args.paper_output_dir is not None:
        publish_manuscript_package(output, args.selection, args.paper_output_dir)
    try:
        display = output.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        display = output.resolve().as_posix()
    print(display)


if __name__ == "__main__":
    main()
