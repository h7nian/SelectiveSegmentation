# Selective Segmentation — Baseline Benchmark Report (archived snapshot)

> **ARCHIVED / NON-AUTHORITATIVE.** This report captures the initial 2026-07-06
> benchmark and is not evidence for the current manuscript, release status, or
> scheduler policy. Use the
> [completion audit](output/COMPLETION_AUDIT.md), [active plan](docs/PLAN.md),
> and [repository README](README.md) for current authoritative information.
> The historical measurements below are retained for provenance.

Date: 2026-07-06 | UMN MSI, `saffo-a100`, one A100 per job
Python 3.12.4 / torch 2.12.1 / torchvision 0.27.1 / transformers 5.13.0
Code: repository root (historical metrics JSON in `outputs/eval/`, checkpoints
in `outputs/train/`)

## 1. Setting

A 2×2 design: **foundation vs. traditional model**, each **with and without
target-dataset fine-tuning**. No model receives boxes, points, masks, or any
other location guidance at inference.

### 1.1 Model conditions

| Condition | Architecture | Weights |
|---|---|---|
| `clipseg-general` | CLIPSeg (ViT-B/16) | pretrained `CIDAS/clipseg-rd64-refined`, zero-shot |
| `clipseg-target` | CLIPSeg (ViT-B/16) | same + decoder fine-tuned on the target training split (CLIP frozen) |
| `deeplabv3-external` | DeepLabV3-ResNet50 | torchvision COCO checkpoint (`COCO_WITH_VOC_LABELS_V1`), zero-shot |
| `deeplabv3-target` | DeepLabV3-ResNet50 | same + fine-tuned end to end on the target training split |

### 1.2 Datasets

| Dataset | Task | Fine-tuning split | Evaluation split |
|---|---|---|---|
| Oxford-IIIT Pet | binary (background / pet), trimap border ignored | `trainval` (3,680) | `test` (3,669) |
| PASCAL VOC 2012 | 21 classes, 255 ignored | `train` (1,464, no SBD) | `val` (1,449) |

### 1.3 Evaluation protocol

- Forward pass at model resolution (CLIPSeg 352², DeepLabV3 512²);
  probability maps **bilinearly upsampled to the original resolution**
  before metric accumulation.
- CLIPSeg prompts: Pet "cat"/"dog", merged into "pet" by pixelwise max;
  VOC the 20 class names. Background = 1 − max(foreground), i.e. foreground
  needs some prompt above 0.5.
- DeepLabV3-external on Pet: pet = cat + dog softmax probabilities; on VOC
  the checkpoint vocabulary already matches.
- Ignored pixels are excluded from all losses and metrics.

### 1.4 Metric definitions

Two aggregation levels: class-level scores (§1.4.1) pool pixels per class
over the whole split; image-level scores (§1.4.2) are computed per image,
then averaged. For class $c$: $\mathrm{TP}_c$, $\mathrm{FP}_c$,
$\mathrm{FN}_c$ are pixel-level true/false positives and false negatives,
$n_c$ its ground-truth pixel count, $N$ the total non-ignored pixels.

Provenance: pixel accuracy, mAcc, mIoU, FWIoU are the classic FCN quartet
(Long et al., CVPR 2015); class-level mIoU is the headline metric of VOC,
Cityscapes, and ADE20K. Dice and HD95 (per case) are the standard pair in
medical imaging. ↑ higher is better, ↓ lower is better.

#### 1.4.1 Class-level metrics (dataset-pooled)

One confusion matrix over all non-ignored pixels of the split; large
instances weigh more within a class, and rare classes still get stable
scores. The convention of VOC/Cityscapes/ADE20K.

**Pixel accuracy** ↑ — fraction of correctly labeled pixels:

$$
\mathrm{PixelAcc} = \frac{\sum_{c} \mathrm{TP}_c}{N}
$$

Dominated by large regions (VOC background alone gives >0.9); reported for
completeness only.

**IoU / mean IoU (Jaccard)** ↑ — overlap over union, macro-averaged over
classes present in prediction or ground truth ($\mathcal{C}_{\mathrm{obs}}$;
others reported as `null`):

$$
\mathrm{IoU}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c},
\qquad
\mathrm{mIoU} = \frac{1}{\lvert \mathcal{C}_{\mathrm{obs}} \rvert}
\sum_{c \in \mathcal{C}_{\mathrm{obs}}} \mathrm{IoU}_c
$$

Each class counts equally, hence much harder than pixel accuracy; the
standard headline metric. Insensitive to *where* boundary errors occur.

**Mean class accuracy (mAcc)** ↑ — per-class recall, averaged over classes
with $n_c > 0$:

$$
\mathrm{Acc}_c = \frac{\mathrm{TP}_c}{n_c},
\qquad
\mathrm{mAcc} = \frac{1}{\lvert \{c : n_c > 0\} \rvert} \sum_{c\,:\,n_c > 0} \mathrm{Acc}_c
$$

Ignores false positives (predicting $c$ everywhere gives $\mathrm{Acc}_c=1$);
high mAcc with low mIoU signals over-prediction.

**Frequency-weighted IoU (FWIoU)** ↑ — IoU weighted by ground-truth pixel
share:

$$
\mathrm{FWIoU} = \sum_{c} \frac{n_c}{N}\, \mathrm{IoU}_c
$$

A middle ground between pixel accuracy and mIoU.

**Dice / mean Dice (F1)** ↑ — harmonic mean of precision and recall; a
monotone transform of IoU:

$$
\mathrm{Dice}_c = \frac{2\,\mathrm{TP}_c}{2\,\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}
= \frac{2\,\mathrm{IoU}_c}{1 + \mathrm{IoU}_c} \;\ge\; \mathrm{IoU}_c
$$

Same ranking as IoU per class, more generous values; the medical-imaging
convention. Macro-averaged like mIoU.

#### 1.4.2 Image-level metrics (per image, averaged over images)

Computed independently within each image, then averaged with equal weight
per image — the per-case convention of medical imaging. Measures "how well
is a typical image segmented" and yields a score distribution over images;
noisier
on rare/tiny classes.

**Image-level mIoU / mean Dice** ↑ — the §1.4.1 formulas evaluated within
each image ($\mathcal{C}_i$ = classes present in image $i$), then averaged
over the $M$ images:

$$
\mathrm{mIoU}^{\mathrm{img}} = \frac{1}{M} \sum_{i=1}^{M}
\frac{1}{\lvert \mathcal{C}_i \rvert} \sum_{c \in \mathcal{C}_i} \mathrm{IoU}_{i,c}
$$

Fully ignored images are skipped. Can rank models differently from the
class-level scores: failing on a rare class costs 1/21 of class-level mIoU
but almost nothing at image level.

**HD95 (95th-percentile Hausdorff distance)** ↓ — boundary error in pixels
at native resolution; necessarily image-level. Let $\partial P$,
$\partial G$ be the surface pixel sets of the predicted and ground-truth
masks (mask minus its erosion). The *directed* Hausdorff distance from one
surface to the other is the worst nearest-neighbor distance:

$$
h(\partial P, \partial G) =
\max_{x \in \partial P} \, \min_{y \in \partial G} \lVert x - y \rVert_2
$$

One direction is not enough. If the prediction lies entirely inside the
ground truth but misses a large part of it, every predicted boundary pixel
is still close to the true boundary — $h(\partial P, \partial G)$ is small —
while much of the true boundary is far from the prediction —
$h(\partial G, \partial P)$ is large. The classic *symmetric* Hausdorff
distance therefore takes the worse of the two directions:

$$
\mathrm{HD}(P, G) = \max\big(
h(\partial P, \partial G),\; h(\partial G, \partial P)
\big)
$$

A single outlier pixel sets this maximum, so we report the robust variant
HD95: pool the nearest-neighbor distances from both directions and take
their 95th percentile (the medpy convention), with
$d(x, S) = \min_{y \in S} \lVert x - y \rVert_2$:

$$
\mathrm{HD}_{95}(P, G) = \mathrm{P}_{95}\Big(
\{\, d(x, \partial G) : x \in \partial P \,\}
\;\cup\;
\{\, d(y, \partial P) : y \in \partial G \,\}
\Big)
$$

The percentile resists outlier pixels while still punishing far-away
false-positive blobs and misaligned boundaries — errors IoU barely sees
(§2.3 has an example: same mIoU, ~2.7 px apart on HD95). Foreground classes
only; computed when both masks contain the class, averaged per class over
such images, then macro-averaged. One-sided presence (detection failure,
distance undefined) is tallied separately.

### 1.5 Training configuration (the `-target` conditions)

| | CLIPSeg-target | DeepLabV3-target |
|---|---|---|
| Trainable parameters | decoder only (CLIP frozen) | all |
| Optimizer / lr | AdamW, 1e-4 | SGD momentum 0.9, 5e-3 |
| Schedule / epochs | poly(0.9), 40 epochs | poly(0.9), 40 epochs |
| Batch size | 32 @ 352² | 16 @ 512² |
| Augmentation | random scale 0.5–2.0×, random crop, horizontal flip | same |
| Loss | per-prompt BCE (VOC: one prompt per image, 25% negatives) | CE + 0.4 × aux CE |
| Wall-clock (1× A100) | Pet 8.7 min / VOC 4.6 min | Pet 39.7 min / VOC 17.7 min |

### 1.6 Alternative training configuration ("config B")

One hyperparameter changed per model, retraining the `-target` conditions
(`outputs/train_b/`, `outputs/eval_b/`):

- **DeepLabV3: 80 epochs** (vs. 40) — tests undertraining, especially on
  VOC (~3.7k steps).
- **CLIPSeg: lr 3e-4** (vs. 1e-4) — one step toward the paper's 1e-3.

## 2. Results

"Config A" is the default configuration; zero-shot conditions are unaffected
by training configuration.

### 2.1 Oxford-IIIT Pet (test, 3,669 images)

| Condition | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 (px) ↓ |
|---|---|---|---|---|
| clipseg-general | 0.9561 | 0.9775 | 0.9799 | 14.15 |
| **clipseg-target** | **0.9819** | **0.9909** | **0.9918** | **5.99** |
| deeplabv3-external | 0.9590 | 0.9790 | 0.9812 | 16.77 |
| deeplabv3-target | 0.9750 | 0.9873 | 0.9886 | 9.59 |

### 2.2 PASCAL VOC 2012 (val, 1,449 images)

| Condition | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 (px) ↓ |
|---|---|---|---|---|
| clipseg-general | 0.6537 | 0.7747 | 0.9069 | 44.29 |
| clipseg-target | 0.7383 | 0.8334 | 0.9261 | 35.36 |
| deeplabv3-external | 0.7639 | 0.8548 | 0.9442 | 35.54 |
| **deeplabv3-target** | **0.7922** | **0.8762** | **0.9542** | **34.11** |

### 2.3 Config A vs. config B (fine-tuned conditions)

| Condition | Dataset | Config | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 (px) ↓ |
|---|---|---|---|---|---|---|
| clipseg-target | Pet | A (lr 1e-4) | 0.9819 | 0.9909 | 0.9918 | 5.99 |
| clipseg-target | Pet | B (lr 3e-4) | **0.9828** | 0.9913 | 0.9922 | **5.76** |
| clipseg-target | VOC | A (lr 1e-4) | 0.7383 | 0.8334 | 0.9261 | 35.36 |
| clipseg-target | VOC | B (lr 3e-4) | **0.7715** | **0.8570** | 0.9359 | **33.00** |
| deeplabv3-target | Pet | A (40 ep) | 0.9750 | 0.9873 | 0.9886 | 9.59 |
| deeplabv3-target | Pet | B (80 ep) | 0.9762 | 0.9880 | 0.9891 | 8.98 |
| deeplabv3-target | VOC | A (40 ep) | **0.7922** | **0.8762** | 0.9542 | **34.11** |
| deeplabv3-target | VOC | B (80 ep) | 0.7889 | 0.8743 | 0.9543 | 36.85 |

**Verdict:** the larger CLIPSeg lr clearly helps (+3.3 mIoU on VOC,
0.738 → 0.772; better everywhere else) — 3e-4 is the better default.
Doubling DeepLabV3's epochs changes nothing beyond noise — config A was not
undertrained; 40 epochs stays the default.

### 2.4 Extended metrics (both configs)

Class-level mAcc and FWIoU, and image-level mIoU/mDice (§1.4).

**Pet**

| Condition | mAcc ↑ | FWIoU ↑ | image-level mIoU ↑ | image-level mDice ↑ |
|---|---|---|---|---|
| clipseg-general | 0.9757 | 0.9606 | 0.9636 | 0.9773 |
| clipseg-target (A) | 0.9925 | 0.9837 | 0.9867 | 0.9917 |
| clipseg-target (B) | **0.9930** | **0.9845** | **0.9876** | **0.9922** |
| deeplabv3-external | 0.9797 | 0.9631 | 0.9618 | 0.9751 |
| deeplabv3-target (A) | 0.9895 | 0.9775 | 0.9800 | 0.9878 |
| deeplabv3-target (B) | 0.9901 | 0.9786 | 0.9812 | 0.9885 |

**VOC 2012**

| Condition | mAcc ↑ | FWIoU ↑ | image-level mIoU ↑ | image-level mDice ↑ |
|---|---|---|---|---|
| clipseg-general | 0.8154 | 0.8450 | 0.6029 | 0.6453 |
| clipseg-target (A) | 0.8775 | 0.8803 | 0.6787 | 0.7142 |
| clipseg-target (B) | **0.8972** | 0.8949 | 0.7114 | 0.7463 |
| deeplabv3-external | 0.8652 | 0.9014 | 0.7748 | 0.8097 |
| deeplabv3-target (A) | 0.8708 | **0.9172** | **0.7845** | **0.8167** |
| deeplabv3-target (B) | 0.8661 | 0.9167 | 0.7824 | 0.8148 |

Two aggregation effects worth noting on VOC: CLIPSeg's image-level mIoU is
*below* its class-level mIoU (0.603 vs. 0.654) while DeepLabV3's is *above*
(0.775 vs. 0.764) — the two families distribute their errors differently
across images. And clipseg-general's FWIoU (0.845) far exceeds its mIoU
(0.654): its weak classes are the rare ones, which FWIoU down-weights.

### 2.5 Per-class IoU — Pet (config A)

| Class | clipseg-general | clipseg-target | deeplabv3-external | deeplabv3-target |
|---|---|---|---|---|
| background | 0.9701 | 0.9876 | 0.9718 | 0.9828 |
| pet | 0.9421 | 0.9763 | 0.9462 | 0.9673 |

### 2.6 Per-class IoU — VOC 2012 (config A)

| Class | clipseg-general | clipseg-target | deeplabv3-external | deeplabv3-target |
|---|---|---|---|---|
| background | 0.9043 | 0.9166 | 0.9363 | 0.9517 |
| aeroplane | 0.8349 | 0.8790 | 0.9301 | 0.9146 |
| bicycle | 0.3425 | 0.4447 | 0.4158 | 0.6675 |
| bird | 0.8561 | 0.9344 | 0.8819 | 0.8588 |
| boat | 0.6951 | 0.8092 | 0.7119 | 0.7373 |
| bottle | 0.7429 | 0.7879 | 0.5614 | 0.7304 |
| bus | 0.8865 | 0.8975 | 0.9527 | 0.9509 |
| car | 0.7259 | 0.8051 | 0.6652 | 0.8881 |
| cat | 0.8341 | 0.8859 | 0.9191 | 0.9097 |
| chair | 0.2787 | 0.2842 | 0.4711 | 0.4327 |
| cow | 0.5621 | 0.8333 | 0.8827 | 0.8604 |
| diningtable | 0.4344 | 0.4665 | 0.5366 | 0.6104 |
| dog | 0.7512 | 0.8381 | 0.8668 | 0.8606 |
| horse | 0.5918 | 0.7924 | 0.8377 | 0.8363 |
| motorbike | 0.7031 | 0.8048 | 0.8784 | 0.8461 |
| person | 0.6936 | 0.8924 | 0.8926 | 0.8881 |
| pottedplant | 0.4182 | 0.5317 | 0.6023 | 0.6109 |
| sheep | 0.5635 | 0.7940 | 0.8979 | 0.8766 |
| sofa | 0.4363 | 0.3823 | 0.5576 | 0.5196 |
| train | 0.8511 | 0.8614 | 0.8668 | 0.9045 |
| tvmonitor | 0.6224 | 0.6623 | 0.7765 | 0.7812 |
