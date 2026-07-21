"""CPU checks for the SDC coupling example and the current exact knot sum."""

from itertools import product
from math import hypot

import numpy as np
from scipy import ndimage


def dice(state, yhat):
    state = np.asarray(state, dtype=bool)
    yhat = np.asarray(yhat, dtype=bool)
    denom = state.sum() + yhat.sum()
    return 0.0 if denom == 0 else 2.0 * np.logical_and(state, yhat).sum() / denom


def coupling_report(name, law, yhat):
    e_dice = 0.0
    e_num = 0.0
    e_den = 0.0
    for state, probability in law:
        state = np.asarray(state, dtype=int)
        numerator = 2 * np.dot(state, yhat)
        denominator = state.sum() + yhat.sum()
        e_dice += probability * dice(state, yhat)
        e_num += probability * numerator
        e_den += probability * denominator
    print(
        f"{name:11s}: E[Dice]={e_dice:.12f}, "
        f"E[num]/E[den]={e_num / e_den:.12f}, "
        f"E[num]={e_num:.3f}, E[den]={e_den:.3f}"
    )


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95_with_empty_penalty(mask, target):
    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target, dtype=bool)
    diagonal = hypot(*mask.shape)
    if not mask.any() or not target.any():
        return diagonal
    mask_surface = surface(mask)
    target_surface = surface(target)
    to_target = ndimage.distance_transform_edt(~target_surface)[mask_surface]
    to_mask = ndimage.distance_transform_edt(~mask_surface)[target_surface]
    return float(np.percentile(np.hstack([to_target, to_mask]), 95))


def main():
    print("SDC COUPLING EXAMPLE")
    yhat = np.array([1, 0])
    independent = [
        (state, 0.25)
        for state in product((0, 1), repeat=2)
    ]
    # With p=(1/2,1/2), a common threshold gives 11 half the time and 00 half.
    comonotone = [((0, 0), 0.5), ((1, 1), 0.5)]
    coupling_report("independent", independent, yhat)
    coupling_report("Q_p", comonotone, yhat)
    p = np.array([0.5, 0.5])
    sdc = 2 * np.dot(p, yhat) / (p.sum() + yhat.sum())
    print(f"SDC        : {sdc:.12f}\n")
    assert np.isclose(sdc, 0.5)

    print("EXACT KNOT SUM")
    pmap = np.array(
        [
            [.1, .1, .1, .1, .1],
            [.1, .3, .3, .3, .1],
            [.1, .3, .9, .3, .1],
            [.1, .3, .3, .3, .1],
            [.1, .1, .1, .1, .1],
        ]
    )
    yhat2 = pmap >= 0.3

    h_cache = {}

    def H(t):
        level = pmap >= t
        key = level.tobytes()
        if key not in h_cache:
            h_cache[key] = hd95_with_empty_penalty(level, yhat2)
        return h_cache[key]

    # Include 0 and 1 to expose both tails. The historical left-endpoint formula
    # was wrong for closed level sets. The current appendix uses the right value
    # on each interval (equivalently, the probability gap below each knot).
    knots = np.r_[0.0, np.unique(pmap), 1.0]
    historical_wrong_left = sum(
        (b - a) * H(a) for a, b in zip(knots[:-1], knots[1:])
    )
    current_exact = sum(
        (b - a) * H(b) for a, b in zip(knots[:-1], knots[1:])
    )
    grid = (np.arange(200_000) + 0.5) / 200_000
    numeric = np.mean([H(t) for t in grid])
    print("knots      :", knots)
    print("H(knots)   :", np.array([H(t) for t in knots]))
    print(f"historical wrong-left: {historical_wrong_left:.12f}")
    print(f"current exact        : {current_exact:.12f}")
    print(f"dense-mid  : {numeric:.12f}")
    assert not np.isclose(historical_wrong_left, current_exact)
    assert np.isclose(current_exact, numeric, rtol=0.0, atol=1e-10)

    rng = np.random.default_rng(7)
    thresholds = rng.random(200_000)
    empty_frequency = np.mean(thresholds > pmap.max())
    print(
        f"empty mass : analytic={1 - pmap.max():.6f}, "
        f"Monte Carlo={empty_frequency:.6f}"
    )
    p_with_one = pmap.copy()
    p_with_one[2, 2] = 1.0
    print(f"max=1 case : empty mass={1 - p_with_one.max():.6f}")


if __name__ == "__main__":
    main()
