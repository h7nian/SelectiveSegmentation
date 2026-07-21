"""Evaluate one benchmark condition on a dataset's held-out split.

Without --checkpoint this evaluates the pretrained model (clipseg-general /
deeplabv3-external); with --checkpoint it evaluates the fine-tuned one
(clipseg-target / deeplabv3-target)::

    python -m selectseg.evaluate --model clipseg --dataset pet

Predictions are upsampled to each image's original resolution before
computing metrics.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selectseg.data import SPECS, SegDataset, eval_collate
from selectseg.metrics import ConfusionMatrix, HausdorffMetric, PerImageMetrics
from selectseg.models import CONDITION_NAMES, build_model

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CONDITION_NAMES))
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--checkpoint", default=None, help="fine-tuned checkpoint to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval",
        help="directory for the metrics JSON",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap images (smoke testing)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    finetuned = args.checkpoint is not None
    condition = CONDITION_NAMES[args.model][finetuned]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SPECS[args.dataset]

    model = build_model(
        args.model, spec, finetuned=finetuned, init_weights=not finetuned
    )
    if finetuned:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.to(device).eval()

    dataset = SegDataset(
        spec, args.data_root, train=False, image_size=model.image_size
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collate,
        pin_memory=device.type == "cuda",
    )

    confusion = ConfusionMatrix(spec.class_names)
    hausdorff = HausdorffMetric(spec.class_names)
    per_image = PerImageMetrics(spec.class_names)
    with torch.inference_mode():
        for images, masks in tqdm(loader, desc=f"{condition} on {args.dataset}"):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                probs = model.predict_probs(images.to(device, non_blocking=True))
            probs = probs.float()
            for prob, mask in zip(probs, masks):
                upsampled = F.interpolate(
                    prob.unsqueeze(0),
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                prediction = upsampled.argmax(dim=0).cpu()
                confusion.update(prediction, mask)
                hausdorff.update(prediction, mask)
                per_image.update(prediction, mask)

    metrics = {**confusion.compute(), **hausdorff.compute(), **per_image.compute()}
    result = {
        "condition": condition,
        "model": args.model,
        "dataset": args.dataset,
        "split": spec.eval_split,
        "checkpoint": args.checkpoint,
        "num_images": len(dataset),
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{condition}_{args.dataset}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"condition      {condition}")
    print(f"dataset        {args.dataset} ({spec.eval_split}, {len(dataset)} images)")
    print(f"pixel accuracy {metrics['pixel_accuracy']:.4f}")
    print(f"mean IoU       {metrics['mean_iou']:.4f}")
    print(f"mean Dice      {metrics['mean_dice']:.4f}")
    print(f"mean class acc {metrics['mean_class_accuracy']:.4f}")
    print(f"fw IoU         {metrics['fw_iou']:.4f}")
    per_image_iou = metrics["per_image_mean_iou"]
    print(
        "per-image mIoU "
        + ("n/a" if per_image_iou is None else f"{per_image_iou:.4f}")
    )
    mean_hd95 = metrics["mean_hd95"]
    print(f"mean HD95      {'n/a' if mean_hd95 is None else f'{mean_hd95:.2f} px'}")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
