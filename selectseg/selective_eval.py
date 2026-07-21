"""Per-image confidence scores and quality metrics for one condition.

Writes one JSON line per evaluation image, pairing the image-level
confidence scores from :mod:`selectseg.selective` with the image's true
quality (mIoU, mean Dice, HD95), for the analysis in
scripts/analyze_selective.py::

    python -m selectseg.selective_eval --model clipseg --dataset pet
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selectseg.data import SPECS, SegDataset, eval_collate
from selectseg.evaluate import CONDITION_NAMES
from selectseg.metrics import ConfusionMatrix, hausdorff_95
from selectseg.models import build_model
from selectseg.selective import image_confidence_scores


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
        default="outputs/selective",
        help="directory for the per-image JSONL",
    )
    parser.add_argument(
        # the sweep is bracketed on both sides: 0.01 is degenerate for diffuse
        # (zero-shot) maps, whose conservative set {p >= 1 - alpha} is empty,
        # and the previous grid stopped at 0.1 while still improving, so the
        # optimum was never enclosed.
        "--alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap images (smoke testing)"
    )
    return parser.parse_args()


def posterior_assumption_diagnostics(probs, mask, spec):
    """How badly do the two rival posterior assumptions fail on this image?

    Both our level-set posterior and SDC's marginal one are *choices* of a joint
    distribution consistent with the model's per-pixel marginals -- a probability
    map does not determine which pixels flip *together*. Neither choice is
    derived, and the error they induce is the one term of the risk decomposition
    that no amount of quadrature can reduce.

    BOTH STATISTICS ARE CORRECTLY IMPLEMENTED AND NEITHER IDENTIFIES WHAT AN
    EARLIER VERSION OF THIS DOCSTRING CLAIMED. It said these two numbers turn that
    term "from an excuse into a number", and that we expect SDC's assumption to be
    "grossly violated everywhere". Both readings are RETRACTED. The failure is
    IDENTIFICATION, not power, and it is not fixable by computing either statistic
    more carefully:

    * Each tests a CONJUNCTION. The residual decomposes as (Y* - q) + (q - p) for
      the true marginal q, so each statistic tests "coupling AND p = q" and NEITHER
      can attribute a violation to the coupling leg.
    * Data generated under SDC's OWN assumption reproduces the entire measured
      signature. Y* ~ indep Bern(q) for a deterministic mask q, with p = blur(q)
      (accurate but MIScalibrated), yields auc ~ 0.9995, moran ~ 0.84-0.95, and the
      same correlation signs against image quality -- while satisfying conditional
      independence EXACTLY. So "moran ~ 0.9 vs ~0 under independence" does NOT
      refute independence: the ~0 null silently assumes perfect calibration.
    * On single-annotation data (Pet/VOC) the question is VACUOUS: the true marginal
      is near-deterministic, so all couplings coincide. What these two statistics
      empirically track is MODEL ACCURACY -- rho(auc, image_miou) = +0.41..+0.85,
      rho(moran, image_miou) = -0.28..-0.77.

    Deciding the coupling needs MULTIPLE ANNOTATIONS per image (LIDC-IDRI, QUBIQ):
    estimate q_hat = mean over raters and test the foreground-count spread, where
    independence gives Var(|Y|) = sum q(1-q) and Q_p gives an sd 23-75x larger.
    That is a precondition for the claim, not a coverage gap. These diagnostics are
    still worth writing out -- they are cheap and they are what the multi-rater
    version of the test would be built on -- but they do not decide the assumption.

    ``auc`` takes the true mask to be a superlevel set of the probability map,
    Y* = {i : p_i >= T}: the model is assumed to *rank* pixels correctly by
    foregroundness, with only the cut-point unknown. That is exactly the statement
    that p, read as a per-pixel classifier of Y*, separates the two classes
    perfectly, so the within-image AUC is 1 iff that holds. 1 - AUC is the fraction
    of discordant (foreground, background) pairs -- a violation rate ONLY given
    p = q.

    ``moran`` is Moran's I (rook adjacency) of the residual Y* - p, which is ~0
    under "conditional independence AND perfect calibration" and positive under the
    spatial coherence that segmentation masks manifestly have. The two legs are not
    separable here; see above.

    What a violation would cost SDC is a SEPARATE question -- and the framing above
    invented a non-tension, because SDC's value is a function of (p, yhat) ALONE, so
    no coupling diagnostic can move it. SDC is not SURVIVING a violated assumption;
    it is INVARIANT to it. But the tempting REASON for that invariance is wrong. An
    earlier version of this docstring said the violation would "cost SDC very little
    anyway, because Dice is a permutation-invariant functional and cannot see the
    coupling at all". That is REFUTED, twice over:

    * SDC is coupling-free because of WHAT IT COMPUTES, not because Dice is blind to
      the coupling. SDC is the RATIO-OF-EXPECTATIONS instantiation of Dice (prop:sdc),
      2 sum_i p_i yhat_i / (sum_i p_i + |Yhat|) -- a functional of the MARGINALS alone,
      so every coupling consistent with those marginals gives it the same value.
      Expected Dice E_Q[Dice(Y, Yhat)] is a DIFFERENT object: an expectation of a
      RATIO, i.e. a nonlinear function of two linear statistics, whose joint law DOES
      depend on the coupling (rem:notcoupling). For N = 2, p = (1/2, 1/2),
      Yhat = {1}: independence gives 5/12, the comonotone Q_p gives 1/3, and SDC gives
      1/2. Three distinct values. PERMUTATION invariance (lem:perm) is not COUPLING
      invariance, and inferring the second from the first is the single most tempting
      error in this area.
    * Nor does SDC's own error bound transport. Its <1% relative-error bound is
      proved under conditional independence and does NOT carry over to Q_p, and the
      bound cannot be rescued by taking foreground mass large: Q_p is COMONOTONE, so
      Var(sum_i Y_i) sits at the Frechet-Hoeffding maximum and the delta-method step
      never engages at any mass (main.tex, rem:notcoupling). That is a theorem, not
      a sweep.
      RETRACTED HERE: this bullet used to quantify the Q_p gap as "4.7% mean / 26%
      max", stated as MEASURED and without main.tex's "on synthetic maps" qualifier.
      No code ever computed it, and the number is UNIDENTIFIABLE rather than merely
      undocumented -- across defensible synthetic families the mean spans
      0.21-11.37% and the max 0.38-97.45%, so the number IS the generator. It is
      withdrawn permanently; do not reintroduce a figure here. The qualitative claim
      (the bound does not transport) is unaffected and is what the argument uses.

    What lem:perm actually gives is the weaker, load-bearing statement: SDC,
    E_{Q_p}[Dice] and the AREA readouts are PERMUTATION-invariant, so no such
    functional can resolve boundary geometry, while the DISTANCE readout is not.
    Reporting both diagnostics is what makes that asymmetry visible.

    Averaged over the foreground classes present in the ground truth; ``None``
    when no such class exists.
    """
    valid = (mask >= 0) & (mask < spec.num_classes)
    aucs, morans = [], []
    for index in range(1, spec.num_classes):
        truth = ((mask == index) & valid).numpy().ravel()
        keep = valid.numpy().ravel()
        if not truth.any() or truth[keep].all():
            continue  # AUC undefined without both classes present
        prob = probs[index].numpy().ravel()
        foreground = keep & truth.astype(bool)
        positive, negative = prob[foreground], prob[keep & ~truth.astype(bool)]
        # Mann-Whitney U / rank-sum form of AUC: O(N log N), no pairwise blowup.
        # Ties MUST get averaged ranks: probability maps are full of plateaus
        # (saturated 0s and 1s), and breaking ties by position would score an
        # all-tied map as AUC 0 or 1 rather than the correct 0.5.
        ranks = stats.rankdata(np.concatenate([positive, negative]))
        rank_sum = ranks[: positive.size].sum()
        aucs.append(
            float(
                (rank_sum - positive.size * (positive.size + 1) / 2)
                / (positive.size * negative.size)
            )
        )
        # Moran's I of the residual, rook adjacency, over the VALID pixels only.
        #
        # The exclusion is not cosmetic. `& valid` alone would only zero the
        # *truth* at an ignored pixel while keeping its probability, so a void
        # pixel would enter the statistic as a fabricated (0 - p) false-positive
        # residual. VOC's void label (255) and Pet's trimap border are not
        # scattered noise: they form a contiguous ring around every object
        # boundary -- exactly where p is most uncertain and the residual is
        # largest -- so they would inject a large *coherent* structure into the
        # very statistic that exists to measure the residual's coherence. The
        # error grows with model quality (a sharp model's valid residual is ~0, so
        # the fabricated one dominates) and can invert the per-image ranking. The
        # AUC above already restricts to `keep`; this now does too.
        residual = ((mask == index) & valid).float().numpy() - probs[index].numpy()
        ok = valid.numpy()
        z = residual - residual[ok].mean()
        denominator = float((z[ok] ** 2).sum())
        if denominator <= 0:
            continue
        # a rook pair contributes only if BOTH of its ends are valid
        horizontal = ok[:, :-1] & ok[:, 1:]
        vertical = ok[:-1, :] & ok[1:, :]
        cross = float(
            (z[:, :-1] * z[:, 1:])[horizontal].sum()
            + (z[:-1, :] * z[1:, :])[vertical].sum()
        )
        pairs = int(horizontal.sum() + vertical.sum())
        if not pairs:
            continue
        morans.append(int(ok.sum()) * cross / (pairs * denominator))
    if not aucs:
        return {"levelset_auc": None, "residual_moran_i": None}
    return {
        "levelset_auc": sum(aucs) / len(aucs),
        "residual_moran_i": sum(morans) / len(morans) if morans else None,
    }


def image_quality(prediction, mask, spec):
    """True per-image quality: mIoU, mean Dice, and mean HD95.

    HD95 is undefined for a class present on only one side (a hallucinated or
    a wholly missed class), so two conventions are reported. ``image_hd95``
    averages over the classes present in *both* the prediction and the ground
    truth, and is ``None`` when no class qualifies -- the convention of the
    medical-imaging literature, but it costs a detection failure *nothing* and
    silently deletes the images that fail outright. ``image_hd95_penalized``
    charges every one-sided class the image diagonal, the largest boundary
    error the image admits, so that misses are counted as the worst errors
    they are and every risk is evaluated on the same image set. Both are
    written; the analysis reports both.

    ``image_hd95_penalized`` is therefore never ``None``: an image with no
    foreground class on *either* side has no boundary to get wrong (it scores
    mIoU = mDice = 1.0), so its penalized boundary error is 0.0, not undefined.
    Returning ``None`` there silently poisoned the whole hd95_penalized column --
    numpy coerces None to NaN, every AURC under that risk becomes NaN, and the
    band-vs-SDC bootstrap (whose NaN comparisons are all False, so the poisoned
    resamples score as losses) reports a plausible but WRONG number instead of
    announcing itself. 0.0 is both the honest value and the one that keeps this
    risk defined on the same image set as the overlap risks, which is the only
    reason it exists. It never fired on the released splits -- every VOC and Pet
    ground truth contains an object -- but a negative/healthy control image is
    routine in the medical-imaging setting whose HD95 convention this follows.
    """
    confusion = ConfusionMatrix(spec.class_names)
    confusion.update(prediction, mask)
    if confusion.matrix.sum() == 0:
        return None
    scores = confusion.compute()
    valid = (mask >= 0) & (mask < spec.num_classes)
    diagonal = math.hypot(*mask.shape[-2:])
    distances, penalized = [], []
    for index in range(1, spec.num_classes):
        predicted = (prediction == index) & valid
        truth = (mask == index) & valid
        if not (predicted.any() or truth.any()):
            continue
        distance = hausdorff_95(predicted, truth)
        if distance is not None:
            distances.append(distance)
            penalized.append(distance)
        else:
            # present on exactly one side: a detection failure
            penalized.append(diagonal)
    return {
        "image_miou": scores["mean_iou"],
        "image_mdice": scores["mean_dice"],
        "image_hd95": sum(distances) / len(distances) if distances else None,
        # never None: an empty-vs-empty image has zero boundary error (see above)
        "image_hd95_penalized": (
            sum(penalized) / len(penalized) if penalized else 0.0
        ),
    }


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{condition}_{args.dataset}.jsonl"

    index = 0
    with output_path.open("w") as output, torch.inference_mode():
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
                ).squeeze(0).cpu()
                quality = image_quality(upsampled.argmax(dim=0), mask, spec)
                if quality is not None:
                    record = {
                        "index": index,
                        **quality,
                        **posterior_assumption_diagnostics(upsampled, mask, spec),
                        **image_confidence_scores(upsampled, args.alphas),
                    }
                    output.write(json.dumps(record) + "\n")
                index += 1

    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
