"""CPU-only counterexamples to docs/main.tex Proposition ``vshape``.

Run from the repository root with::

    .venv/bin/python scratch_attack/claim1_vshape_counterexamples.py

The HD95 implementation below is the same pooled, bidirectional surface-distance
percentile used in ``selectseg.metrics.hausdorff_95``.  As in the proposition's
bounded-loss convention, an empty level set is charged the image diagonal.
"""

import math

import numpy as np
from scipy import ndimage


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95(level, prediction):
    if not level.any() or not prediction.any():
        return math.hypot(*level.shape)
    level_surface = surface(level)
    prediction_surface = surface(prediction)
    level_to_prediction = ndimage.distance_transform_edt(~prediction_surface)[
        level_surface
    ]
    prediction_to_level = ndimage.distance_transform_edt(~level_surface)[
        prediction_surface
    ]
    return float(
        np.percentile(
            np.hstack([level_to_prediction, prediction_to_level]), 95
        )
    )


def height(probability, prediction, threshold):
    return hd95(probability >= threshold, prediction)


print("A. Binary softmax: H is not V-shaped")
h = w = 31
p_class = np.full((h, w), 0.05)
row, col = np.ogrid[:h, :w]
disk = (row - 15) ** 2 + (col - 6) ** 2 <= 4**2
p_class[disk] = 0.20       # many low-distance boundary samples
p_class[15, 28] = 0.40    # one remote, higher-confidence false component
p_class[15, 6] = 0.90     # the sole argmax-positive pixel
p_binary = np.stack([1 - p_class, p_class])
assert np.allclose(p_binary.sum(axis=0), 1)
prediction = p_binary.argmax(axis=0) == 1
thresholds = [0.01, 0.10, 0.30, 0.45, 0.50, 0.95]
heights = [height(p_class, prediction, t) for t in thresholds]
for t, value in zip(thresholds, heights):
    print(f"  t={t:>4.2f}: |Y_t|={(p_class >= t).sum():>3}, H={value:.6f}")
assert heights[1] < heights[2] and heights[2] > heights[3]
assert heights[3] == heights[4] == 0


print("\nB. A 3-way softmax: 1/C is not an HD95 minimizer")
h = w = 11
p = np.empty((3, h, w))
p[0], p[1], p[2] = 0.34, 0.50, 0.16
central = np.zeros((h, w), dtype=bool)
central[4:7, 4:7] = True
p[0, central], p[1, central], p[2, central] = 0.40, 0.35, 0.25
assert np.allclose(p.sum(axis=0), 1)
prediction = p.argmax(axis=0) == 0
h_floor = height(p[0], prediction, 1 / 3)
h_above = height(p[0], prediction, 0.35)
print(f"  column sums: [{p.sum(axis=0).min():.1f}, {p.sum(axis=0).max():.1f}]")
print(f"  min observed winning p_c: {p[0, prediction].min():.2f}")
print(f"  H(1/3)={h_floor:.6f}; H(0.35)={h_above:.6f}")
assert h_floor > 0 and h_above == 0

print("  Nor must the minimizer lie above 1/C:")
h = w = 31
p = np.empty((3, h, w))
p[0], p[1], p[2] = 0.05, 0.90, 0.05
row, col = np.ogrid[:h, :w]
disk = (row - 15) ** 2 + (col - 6) ** 2 <= 4**2
p[0, disk], p[1, disk], p[2, disk] = 0.20, 0.60, 0.20
p[:, 15, 6] = [0.35, 0.34, 0.31]   # c wins, barely
p[:, 15, 28] = [0.40, 0.45, 0.15]  # c loses despite a larger p_c
assert np.allclose(p.sum(axis=0), 1)
prediction = p.argmax(axis=0) == 0
below_floor = height(p[0], prediction, 0.10)
at_floor = height(p[0], prediction, 1 / 3)
above_floor = height(p[0], prediction, 0.36)
print(
    f"    H(0.10)={below_floor:.6f}; H(1/3)={at_floor:.6f}; "
    f"H(0.36)={above_floor:.6f}"
)
assert below_floor < at_floor < above_floor


print("\nC. C>2 can still have H(1/C)=0")
h = w = 11
central = np.zeros((h, w), dtype=bool)
central[4:7, 4:7] = True
p = np.empty((3, h, w))
p[0], p[1], p[2] = 0.10, 0.80, 0.10
p[0, central], p[1, central], p[2, central] = 0.80, 0.10, 0.10
assert np.allclose(p.sum(axis=0), 1)
prediction = p.argmax(axis=0) == 0
h_floor_zero = height(p[0], prediction, 1 / 3)
print(f"  Y_(1/3) equals Yhat: {np.array_equal(p[0] >= 1 / 3, prediction)}")
print(f"  H(1/3)={h_floor_zero:.6f}")
assert h_floor_zero == 0

print("  It can even be zero when the two masks are unequal:")
h = w = 25
p = np.empty((3, h, w))
p[0], p[1], p[2] = 0.10, 0.80, 0.10
large_central = np.zeros((h, w), dtype=bool)
large_central[8:17, 8:17] = True
p[0, large_central], p[1, large_central], p[2, large_central] = 0.40, 0.35, 0.25
p[:, 2, 2] = [0.34, 0.50, 0.16]  # one losing p_c >= 1/3 pixel
assert np.allclose(p.sum(axis=0), 1)
prediction = p.argmax(axis=0) == 0
level = p[0] >= 1 / 3
h_robust_zero = hd95(level, prediction)
pooled_surface_samples = int(surface(level).sum() + surface(prediction).sum())
print(
    f"    equal={np.array_equal(level, prediction)}, "
    f"symmetric difference={(level ^ prediction).sum()}, "
    f"pooled surface samples={pooled_surface_samples}, H(1/3)={h_robust_zero:.6f}"
)
assert not np.array_equal(level, prediction) and h_robust_zero == 0

print("  Conversely, C=2 is not sufficient without a no-ties assumption:")
h = w = 11
p_class = np.full((h, w), 0.50)
p_class[4:7, 4:7] = 0.80
p_binary = np.stack([1 - p_class, p_class])
prediction = p_binary.argmax(axis=0) == 1  # NumPy sends exact ties to class 0
h_tied = height(p_class, prediction, 0.50)
print(f"    |Yhat|={prediction.sum()}, |Y_(1/2)|={(p_class >= .5).sum()}, H(1/2)={h_tied:.6f}")
assert h_tied > 0


print("\nD. CLIPSeg-style: 1/2 is only a winner lower bound, not a minimizer")
h = w = 11
central = np.zeros((h, w), dtype=bool)
central[4:7, 4:7] = True
foreground = np.empty((2, h, w))
foreground[0], foreground[1] = 0.60, 0.90
foreground[0, central], foreground[1, central] = 0.80, 0.70
background = 1 - foreground.max(axis=0)
p_clipseg = np.concatenate([background[None], foreground], axis=0)
prediction = p_clipseg.argmax(axis=0) == 1
h_half = height(foreground[0], prediction, 0.50)
h_seven = height(foreground[0], prediction, 0.70)
print(f"  H(0.50)={h_half:.6f}; H(0.70)={h_seven:.6f}")
assert h_half > 0 and h_seven == 0
