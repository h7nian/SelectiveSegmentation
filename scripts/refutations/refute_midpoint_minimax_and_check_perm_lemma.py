"""CPU-only numerical checks for theory claims (2) and (3)."""

import numpy as np
from scipy.integrate import quad
from scipy.ndimage import binary_erosion, distance_transform_edt


def star_discrepancy(nodes):
    x = np.sort(np.asarray(nodes, dtype=float))
    m = x.size
    return float(
        np.max(
            np.r_[
                x - np.arange(m) / m,
                np.arange(1, m + 1) / m - x,
            ]
        )
    )


def v_step(t):
    """A bounded-variation, piecewise-constant V with V(f)=6."""
    d = min(t, 1 - t)
    if d < 0.10:
        return 3.0
    if d < 0.20:
        return 2.0
    if d < 0.45:
        return 1.0
    return 0.0


def dice(a, b):
    denominator = a.sum() + b.sum()
    return 1.0 if denominator == 0 else 2 * np.logical_and(a, b).sum() / denominator


def iou(a, b):
    denominator = np.logical_or(a, b).sum()
    return 1.0 if denominator == 0 else np.logical_and(a, b).sum() / denominator


def sdc(p, prediction):
    return 2 * (p * prediction).sum() / (p.sum() + prediction.sum())


def expected_level_set_dice(p, prediction):
    cuts = np.unique(np.r_[0.0, p.ravel(), 1.0])
    return sum(
        (b - a) * dice(p >= (a + b) / 2, prediction)
        for a, b in zip(cuts[:-1], cuts[1:])
    )


def surface(mask):
    return mask & ~binary_erosion(mask, border_value=0)


def hd95(a, b):
    sa, sb = surface(a), surface(b)
    a_to_b = distance_transform_edt(~sb)[sa]
    b_to_a = distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.r_[a_to_b, b_to_a], 95))


def mean_iou(a, b):
    values = []
    for c in (False, True):
        aa, bb = a == c, b == c
        union = np.logical_or(aa, bb).sum()
        if union:
            values.append(np.logical_and(aa, bb).sum() / union)
    return float(np.mean(values))


print("CLAIM 2")
integral = quad(v_step, 0, 1, points=[.10, .20, .45, .55, .80, .90])[0]
variation = 6.0
for name, nodes in [
    ("ordinary midpoint", [.25, .75]),
    ("actual-error tuned", [.15, .75]),
]:
    estimate = np.mean([v_step(x) for x in nodes])
    discrepancy = star_discrepancy(nodes)
    print(
        f"{name:18s}: I={integral:.1f}, Q={estimate:.1f}, "
        f"error={abs(estimate-integral):.1f}, D*={discrepancy:.2f}, "
        f"V*D*={variation*discrepancy:.1f}"
    )
print(f"D* split at t*=.20: {star_discrepancy([.10, .60]):.3f}")
print(f"D* split at t*=.05: {star_discrepancy([.025, .525]):.3f}\n")


print("CLAIM 3")
n = 32
y, x = np.mgrid[:n, :n]
radius = np.hypot(y - (n - 1) / 2, x - (n - 1) / 2)
p = np.clip(1 - radius / (.6 * n), .02, .98)
prediction = p >= .5
alpha = .30
permutation = np.random.default_rng(20260714).permutation(p.size)
permuted_p = p.ravel()[permutation].reshape(p.shape)
permuted_prediction = prediction.ravel()[permutation].reshape(p.shape)

rows = []
for label, q, h in [
    ("original", p, prediction),
    ("permuted", permuted_p, permuted_prediction),
]:
    low, high = q >= alpha, q >= 1 - alpha
    values = np.array(
        [
            sdc(q, h),
            expected_level_set_dice(q, h),
            dice(high, low),
            iou(high, low),
            hd95(low, high),
        ]
    )
    rows.append(values)
    print(label, *(f"{v:.15f}" for v in values))
    print("  areas/surfaces:", low.sum(), surface(low).sum(), high.sum(), surface(high).sum())

assert np.allclose(rows[0][:4], rows[1][:4], rtol=0, atol=1e-14)
assert rows[0][4] != rows[1][4]

ground_truth = ((y - 17.0) ** 2 + (x - 14.0) ** 2) <= 10.0**2
permuted_ground_truth = ground_truth.ravel()[permutation].reshape(ground_truth.shape)
original_risk = 1 - mean_iou(ground_truth, prediction)
permuted_risk = 1 - mean_iou(permuted_ground_truth, permuted_prediction)
print(f"jointly permuted 1-mIoU: {original_risk:.15f}, {permuted_risk:.15f}")
