"""Freeze one model/dataset condition into an immutable binary-map artifact.

This command performs only model inference and native-resolution alignment.  It
does not choose a hard decision threshold or compute a confidence score; those
simulation-specific choices belong in independent downstream Slurm jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selectseg.artifacts import sha256_file, write_binary_artifact
from selectseg.data import SPECS, SegDataset, eval_collate
from selectseg.models import (
    CLIPSEG_CHECKPOINT,
    CONDITION_NAMES,
    DEEPLABV3_WEIGHTS,
    SEGFORMER_CHECKPOINT,
    build_model,
)
from selectseg.provenance import (
    VerifiedEvalDataset,
    read_verified_sample_file,
    verify_condition_inputs,
)


@dataclass(frozen=True)
class ScientificFileReader:
    """Picklable DataLoader callback that verifies each consumed source file."""

    dataset: VerifiedEvalDataset

    def __call__(self, sample_id, role, path):
        return read_verified_sample_file(self.dataset, sample_id, role, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CONDITION_NAMES))
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/binary_maps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-num-samples", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--campaign-config")
    parser.add_argument("--expected-campaign-config-sha256")
    parser.add_argument("--scientific-input-lock")
    parser.add_argument("--expected-scientific-input-lock-sha256")
    parser.add_argument("--expected-condition-input-sha256")
    return parser.parse_args(argv)


def _validate_args(args) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.expected_num_samples is not None and args.expected_num_samples <= 0:
        raise ValueError("--expected-num-samples must be positive")
    scientific_values = (
        args.campaign_config,
        args.expected_campaign_config_sha256,
        args.scientific_input_lock,
        args.expected_scientific_input_lock_sha256,
        args.expected_condition_input_sha256,
    )
    if any(value is not None for value in scientific_values) and not all(
        value is not None for value in scientific_values
    ):
        raise ValueError(
            "scientific execution requires campaign config/hash, scientific "
            "input lock/hash, and condition-input hash together"
        )


def _source_fingerprint() -> str:
    """Hash every local source file that defines a frozen payload."""

    root = Path(__file__).resolve().parents[2]
    paths = [
        root / path
        for path in (
            "selectseg/__init__.py",
            "selectseg/artifacts.py",
            "selectseg/pipeline/freeze.py",
            "selectseg/provenance.py",
            "selectseg/data.py",
            "selectseg/models.py",
            "scripts/slurm/run.sbatch",
            "scripts/slurm/env.sh",
            "scripts/submit/main.py",
        )
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    working_root = Path.cwd().resolve()
    try:
        return resolved.relative_to(working_root).as_posix()
    except ValueError:
        # The schema forbids absolute provenance paths.  A checkpoint outside the
        # project is still identified by its basename and, critically, its digest.
        return resolved.name


def _package_versions() -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "torchvision", "transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _checkpoint_info(checkpoint):
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"checkpoint is not a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"checkpoint cannot be empty: {path}")
    return {
        "path": _portable_path(path),
        "sha256": sha256_file(path),
        "size_bytes": size,
    }


def _command(argv) -> list[str]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return ["python", "-m", "selectseg.pipeline.freeze", *arguments]


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    spec = SPECS[args.dataset]
    if spec.num_classes != 2:
        raise ValueError(
            "selectseg.pipeline.freeze only supports native binary datasets; "
            f"{args.dataset!r} has {spec.num_classes} classes"
        )
    finetuned = args.checkpoint is not None
    condition = CONDITION_NAMES[args.model][finetuned]
    scientific = None
    verified_file_reader = None
    if args.scientific_input_lock is not None:
        config_path = Path(args.campaign_config)
        if not config_path.is_file() or config_path.is_symlink():
            raise FileNotFoundError(
                f"campaign config is not a regular non-symlink file: {config_path}"
            )
        actual_config_sha256 = sha256_file(config_path)
        if actual_config_sha256 != args.expected_campaign_config_sha256:
            raise ValueError("campaign config SHA-256 differs from the planned job")
        scientific = verify_condition_inputs(
            args.scientific_input_lock,
            dataset=args.dataset,
            model=args.model,
            condition=condition,
            expected_sha256=args.expected_scientific_input_lock_sha256,
            mode="consume",
        )
        if scientific["scientific_input_sha256"] != (
            args.expected_condition_input_sha256
        ):
            raise ValueError("condition scientific-input SHA-256 mismatch")
        locked_checkpoint = scientific["checkpoint"]
        if (locked_checkpoint is None) != (args.checkpoint is None):
            raise ValueError("planned checkpoint state differs from scientific lock")
        if locked_checkpoint is not None:
            if Path(locked_checkpoint["path"]).resolve() != Path(
                args.checkpoint
            ).resolve():
                raise ValueError("planned checkpoint path differs from scientific lock")
        verified_file_reader = ScientificFileReader(scientific["eval_dataset"])

    checkpoint = _checkpoint_info(args.checkpoint)
    source_sha256 = _source_fingerprint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set but CUDA is unavailable")

    model = build_model(
        args.model,
        spec,
        finetuned=finetuned,
        init_weights=not finetuned,
    )
    if finetuned:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.to(device).eval()

    dataset_options = {}
    if verified_file_reader is not None:
        dataset_options["verified_file_reader"] = verified_file_reader
    full_dataset = SegDataset(
        spec,
        args.data_root,
        train=False,
        image_size=model.image_size,
        **dataset_options,
    )
    if (
        args.expected_num_samples is not None
        and len(full_dataset) != args.expected_num_samples
    ):
        raise ValueError(
            f"{args.dataset!r} has {len(full_dataset)} evaluation samples; "
            f"expected {args.expected_num_samples}"
        )
    image_indices = list(range(len(full_dataset)))
    if args.limit is not None:
        image_indices = image_indices[: args.limit]
    if not image_indices:
        raise ValueError(f"{args.dataset!r} evaluation split is empty")
    sample_ids = [full_dataset.sample_id(index) for index in image_indices]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{args.dataset!r} contains duplicate sample identifiers")
    dataset = Subset(full_dataset, image_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collate,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    def frozen_samples():
        image_cursor = 0
        with torch.inference_mode():
            for images, masks in tqdm(
                loader,
                desc=f"freeze {condition} on {args.dataset}",
                disable=not sys.stderr.isatty(),
            ):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    probabilities = model.predict_probs(
                        images.to(device, non_blocking=True)
                    )
                if probabilities.ndim != 4 or probabilities.shape[1] != 2:
                    raise ValueError(
                        "binary model must return probabilities with shape (B, 2, H, W)"
                    )
                probabilities = probabilities.float()
                if probabilities.shape[0] != len(masks):
                    raise ValueError("model batch size does not match collated masks")
                for probability, mask in zip(probabilities, masks):
                    dataset_index = image_indices[image_cursor]
                    expected_id = sample_ids[image_cursor]
                    actual_id = full_dataset.sample_id(dataset_index)
                    if actual_id != expected_id:
                        raise RuntimeError(
                            "dataset sample order changed during inference: "
                            f"expected {expected_id!r}, got {actual_id!r}"
                        )
                    image_cursor += 1
                    if mask.ndim != 2:
                        raise ValueError(
                            f"{args.dataset}/{actual_id} truth mask is not two-dimensional"
                        )
                    unique_labels = set(torch.unique(mask).tolist())
                    if not unique_labels <= {0, 1}:
                        raise ValueError(
                            f"{args.dataset}/{actual_id} is not a total binary mask: "
                            f"labels={sorted(unique_labels)}"
                        )
                    foreground = F.interpolate(
                        probability.unsqueeze(0),
                        size=mask.shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)[1]
                    yield (
                        actual_id,
                        foreground.cpu().numpy(),
                        (mask == 1).to(torch.uint8).cpu().numpy(),
                    )
        if image_cursor != len(image_indices):
            raise RuntimeError(
                f"processed {image_cursor} of {len(image_indices)} declared images"
            )

    manifest_path = write_binary_artifact(
        args.output_dir,
        dataset=args.dataset,
        condition=condition,
        model=args.model,
        split=spec.eval_split,
        class_index=1,
        class_name=spec.class_names[1],
        checkpoint=checkpoint,
        base_model={
            "name": args.model,
            "source": {
                "clipseg": CLIPSEG_CHECKPOINT,
                "deeplabv3": str(DEEPLABV3_WEIGHTS),
                "segformer": SEGFORMER_CHECKPOINT,
            }[args.model],
        },
        source_sha256=source_sha256,
        environment={
            "packages": _package_versions(),
            "device": device.type,
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "autocast_dtype": ("bfloat16" if device.type == "cuda" else "disabled"),
        },
        preprocessing={
            "model_input": "square resize with antialiasing",
            "probability_to_native_mask": (
                "bilinear interpolation with align_corners=False"
            ),
        },
        cohort=(
            "all images from a native binary segmentation split"
            if args.limit is None
            else f"first {args.limit} images from a development-only subset"
        ),
        sample_ids=sample_ids,
        samples=frozen_samples(),
        command=_command(argv),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        scientific_input=(
            None
            if scientific is None
            else {
                **scientific["scientific_input_hashes"],
                "condition_input_sha256": scientific[
                    "scientific_input_sha256"
                ],
            }
        ),
    )
    manifest_sha256 = sha256_file(manifest_path)
    print(f"saved {manifest_path}")
    print(f"manifest_sha256={manifest_sha256}")
    # A machine-readable final line makes scheduler logs easy to index without
    # treating stdout as the provenance authority (campaign locks still rehash).
    print(
        json.dumps(
            {
                "manifest_path": manifest_path.as_posix(),
                "manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return manifest_path


if __name__ == "__main__":
    main()
