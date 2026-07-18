"""Fine-tune a model on a target dataset's training split.

Produces the "-target" conditions of the benchmark::

    python -m selectseg.train --model clipseg --dataset pet
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from selectseg.data import SPECS, SegDataset
from selectseg.models import build_model

# Per-model defaults: CLIPSeg trains its small decoder with AdamW, DeepLabV3
# fine-tunes end to end with SGD (its standard recipe).
TRAIN_DEFAULTS = {
    "clipseg": {"lr": 1e-4, "batch_size": 32},
    "deeplabv3": {"lr": 5e-3, "batch_size": 16},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(TRAIN_DEFAULTS))
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to outputs/train/<model>_<dataset>",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="default depends on --model"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="default depends on --model"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="cap batches per epoch (smoke testing)",
    )
    args = parser.parse_args()
    defaults = TRAIN_DEFAULTS[args.model]
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.lr = args.lr or defaults["lr"]
    args.output_dir = Path(
        args.output_dir or f"outputs/train/{args.model}_{args.dataset}"
    )
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SPECS[args.dataset]

    model = build_model(args.model, spec, finetuned=True).to(device)
    dataset = SegDataset(
        spec, args.data_root, train=True, image_size=model.image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    batches_per_epoch = len(loader)
    if args.limit_batches:
        batches_per_epoch = min(batches_per_epoch, args.limit_batches)
    if batches_per_epoch == 0:
        raise RuntimeError("training split is smaller than one batch")

    optimizer = model.configure_optimizer(args.lr, args.weight_decay)
    total_steps = args.epochs * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1 - step / total_steps) ** 0.9
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) for key, value in vars(args).items()}
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
    history_path = args.output_dir / "history.json"
    history_tmp = history_path.with_suffix(".json.tmp")
    history = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        progress = tqdm(
            loader,
            total=batches_per_epoch,
            desc=f"epoch {epoch}/{args.epochs}",
            disable=not sys.stderr.isatty(),
        )
        for step, (images, masks, prompt_indices) in enumerate(progress):
            if step >= batches_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            prompt_indices = prompt_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = model.compute_loss(images, masks, prompt_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        mean_loss = epoch_loss / batches_per_epoch
        print(
            f"epoch {epoch}/{args.epochs}"
            f" loss {mean_loss:.4f}"
            f" lr {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "mean_training_loss": mean_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        history_tmp.write_text(json.dumps(history, indent=2) + "\n")
        history_tmp.replace(history_path)
        torch.save(model.state_dict(), checkpoint_tmp)
        checkpoint_tmp.replace(checkpoint_path)
    print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
