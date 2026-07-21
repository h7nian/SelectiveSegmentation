"""Is the constraint t_lo + t_hi = 1 costing us anything?

The band's two thresholds are (alpha, 1 - alpha): symmetric about the decision
threshold by construction. The integrand is not. On the BINARY maps this script
generates, H_x(t) = L(Y_t, Yhat) is zero at t = 1/2 (Y_{1/2} IS Yhat, no pixel tying at
exactly 1/2) and grows toward both tails, but the RIGHT tail is far heavier: as t -> 1
the level set empties and HD95 saturates at the image diagonal, while as t -> 0 it
fills the frame and HD95 stays bounded by the mask's distance to the border. Imposing a
symmetry the integrand does not have must cost accuracy.

READ THE RETRACTION IN ``variation_allocated_nodes`` BEFORE REUSING ANY OF THIS. Those
two sentences are properties of THIS script's binary generator, not general facts:
H(1/2) = 0 fails for C > 2 and for a binary map with ties at exactly 1/2, and
H(1) = diag fails on a saturating float32 softmax. And the sqrt(V_L/V_R) allocation
this script derives is named in rem:minimaxfallacy as the SECOND appearance of the
minimax non-sequitur -- it provably minimizes the Koksma bound and provably fails to
reduce the error. This script is kept as the RECORD OF A FAILED DERIVATION.

The question is what to do about it. Freeing the two thresholds (and their weights)
buys accuracy but re-introduces the hyperparameters that taking the nodes from a
quadrature rule had just eliminated -- and the optimal asymmetry is image-dependent,
so a single global choice is a compromise and a per-image choice is the trap that
sank both rank_anchored_threshold and importance_nodes. Simply adding NODES is the
other way to cover both tails, and it costs nothing but distance transforms.

So we measure, against the exact gap-weighted integral over every distinct
probability value:

  * the best SYMMETRIC 2-point rule (one free parameter, oracle-tuned)
  * the best ASYMMETRIC 2-point rule, equal weights (two free parameters, oracle)
  * the best ASYMMETRIC 2-point rule, free weights (three free parameters, oracle)
  * the midpoint rule at M = 2, 4, 8, 16 (ZERO free parameters)

The oracle rules are tuned on the very maps they are scored on, so they are upper
bounds on what any tuning procedure could achieve. If a zero-parameter midpoint rule
matches an oracle-tuned asymmetric one, the answer is "add nodes, do not tune".

Run under SLURM (scripts/slurm/asymmetric.sbatch) -- the oracle grid search is far
too heavy for a login node.
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.selective import quadrature_nodes, quadrature_risks  # noqa: E402

N_MAPS = 40
GRID = np.round(np.arange(0.05, 1.0, 0.05), 3)
WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def variation_allocated_nodes(tensor, predicted, count):
    """Nodes derived from the integrand's own structure. NO free parameter.

    RETRACTED -- ALL THREE OF THE "FACTS" THIS RULE IS BUILT ON ARE REFUTED, AND SO IS
    THE DERIVATION THEY FEED. Kept as the record of a failed derivation, and because
    the rule is still measured below so the failure stays visible.

      (i)   H(1/2) = 0 only for a BINARY map, and only if NO pixel attains exactly 1/2
            (argmax sends ties to background, so Yhat = {p > 1/2} STRICTLY while
            Y_{1/2} = {p >= 1/2} keeps them). For C > 2 a class can win the argmax with
            p well below 1/2, so {p >= 1/2} is a strict SUBSET of Yhat_c and H(1/2) > 0.
      (ii)  H is EMPIRICALLY V-shaped with its minimum near m_c = min in-mask
            probability -- NOT at 1/2, and NOT provably monotone on either side.
            Unimodality is not guaranteed: H can fall, rise and fall again
            (27.3, 4.0, 19.8, 0.0 at t = 0.01, 0.10, 0.30, 0.45). obs:vshape is an
            OBSERVATION; the vertex has no closed form (rem:vshapefails).
      (iii) H(1) = diag only when max_i p_i < 1. A confident float32 softmax saturates
            to exactly 1.0, and then Y_1 = {p >= 1} is NON-empty and H(1) = 0, not diag
            (prop:floor(iii)). V_R = diag is therefore not free and not known a priori.

    AND THE DERIVATION ITSELF IS A NON-SEQUITUR. The sqrt(V_L/V_R) allocation below
    provably minimizes the KOKSMA BOUND and provably fails to reduce the ERROR -- a node
    set with a strictly worse bound can achieve a strictly better error
    (rem:minimaxfallacy names this rule as the second appearance of that fallacy). It is
    measured below and it does not help: against uniform allocation at equal M it wins
    twice, ties twice and loses once, with margins under one point.

    All three premises happen to HOLD on this script's own generator (binary maps
    clipped to [0.01, 0.99], predicted = p >= 0.5), which is why the numbers it reports
    are not wrong -- they are simply not evidence for a general rule. What follows is
    the original derivation, preserved:

    Split the integral at the vertex and each half is MONOTONE, so Koksma's
    inequality applies to each with total variation

        V_L = H(0) - H(1/2) = H(0) = HD95(whole image, Yhat)   [one EDT]
        V_R = H(1) - H(1/2) = diag                             [free, known]

    and error e_L <= V_L / (2 M_L), e_R <= V_R / (2 M_R). Minimising e_L + e_R
    subject to M_L + M_R = M gives the optimal allocation

        M_L / M_R = sqrt(V_L / V_R).

    Because the empty level set saturates at the diagonal, V_R is typically much
    the larger, and the theory therefore says: PUT MORE NODES ON THE HIGH-THRESHOLD
    SIDE. That is the asymmetry -- derived, not tuned. The constraint
    t_lo + t_hi = 1 is a symmetry the integrand does not have.

    Within each half the nodes are the composite midpoint rule, which is
    minimax-optimal for a monotone (hence bounded-variation) integrand -- and
    MINIMAX-OPTIMAL IS NOT OPTIMAL, which is exactly the step that fails
    (rem:minimaxfallacy). Each node carries the width of the subinterval it
    represents. The WEIGHTS are what
    keep this safe: only the node *allocation* adapts to the image, while the
    estimand stays exactly int_0^1 H(t) dt, so the cross-image ranking -- all AURC
    depends on -- is untouched. This is variance reduction, not a change of
    estimator, which is precisely what rank_anchored_threshold and importance_nodes
    got wrong.

    Returns (nodes, weights, M_L, ratio).
    """
    diagonal = math.hypot(*predicted.numpy().shape)
    left = quadrature_risks(tensor, predicted, [0.0])
    v_left = left["hd95"] if left is not None else 0.0
    v_right = diagonal
    share = math.sqrt(max(v_left, 1e-9)) / (
        math.sqrt(max(v_left, 1e-9)) + math.sqrt(v_right)
    )
    m_left = min(max(int(round(count * share)), 1), count - 1)
    m_right = count - m_left
    nodes = [(j + 0.5) / m_left * 0.5 for j in range(m_left)]
    nodes += [0.5 + (j + 0.5) / m_right * 0.5 for j in range(m_right)]
    weights = [0.5 / m_left] * m_left + [0.5 / m_right] * m_right
    return nodes, weights, m_left, v_left / v_right


def synthetic_maps(count, seed=3):
    """Smoothed random fields with a range of sharpnesses, plus their argmax masks."""
    rng = np.random.default_rng(seed)
    maps = []
    while len(maps) < count:
        prob = ndimage.gaussian_filter(rng.random((64, 64)), rng.uniform(5, 14))
        prob = (prob - prob.min()) / (prob.max() - prob.min())
        prob = np.clip(prob, 0.01, 0.99)
        predicted = torch.tensor(prob >= 0.5)
        if predicted.any() and not predicted.all():
            maps.append(
                (prob, torch.tensor(prob, dtype=torch.float32), predicted)
            )
    return maps


def exact_integral(prob, tensor, predicted):
    """int_0^1 H(t) dt by the exact gap-weighted sum over every distinct value.

    H is piecewise constant, jumping only where t crosses one of the map's own
    probabilities, so this sum IS the integral -- no quadrature error at all. The
    edges are anchored at 0 and 1, so the tails are included; the upper one carries
    the atom Q_p places on the empty mask, weight 1 - max_i p_i, charged the diagonal.
    """
    values = np.unique(prob)
    edges = np.concatenate([[0.0], values, [1.0]])
    total = 0.0
    for index, value in enumerate(values):
        risk = quadrature_risks(tensor, predicted, [float(value)])
        height = risk["hd95"] if risk is not None else 0.0
        total += (edges[index + 1] - edges[index]) * height
    return total + (1 - values[-1]) * math.hypot(*prob.shape)


def mean_error(maps, exact, nodes, weights=None):
    """Mean relative error of a quadrature rule against the exact integral."""
    errors = []
    for (_, tensor, predicted), truth in zip(maps, exact):
        risk = quadrature_risks(tensor, predicted, list(nodes), weights=weights)
        errors.append(abs(risk["hd95"] - truth) / truth * 100)
    return float(np.mean(errors))


def check_v_shape(maps):
    """Is H really V-shaped -- zero at 1/2, monotone on each half?

    The whole derivation rests on this. If H is not monotone on each half, its total
    variation is NOT H(0) - H(1/2) and H(1) - H(1/2), Koksma's bound is being fed the
    wrong V, and the sqrt(V_L / V_R) allocation is unjustified. HD95 is a percentile,
    not a metric, so monotonicity is plausible but not guaranteed: a level set can
    lose a connected component and make the percentile jump non-monotonically.
    Measure it rather than assume it.
    """
    grid = np.linspace(0.02, 0.98, 49)
    violations, at_half = [], []
    for _, tensor, predicted in maps:
        heights = []
        for t in grid:
            risk = quadrature_risks(tensor, predicted, [float(t)])
            heights.append(risk["hd95"] if risk is not None else np.nan)
        heights = np.array(heights)
        left = heights[grid < 0.5]
        right = heights[grid > 0.5]
        # left half should be non-increasing, right half non-decreasing
        violations.append(
            (np.diff(left) > 1e-9).mean() if left.size > 1 else 0.0
        )
        violations.append(
            (np.diff(right) < -1e-9).mean() if right.size > 1 else 0.0
        )
        risk = quadrature_risks(tensor, predicted, [0.5])
        at_half.append(risk["hd95"] if risk is not None else np.nan)
    print("V-SHAPE CHECK (the retracted derivation rested on this):")
    print(f"  H(1/2)                        : max over maps = {np.nanmax(at_half):.4f}"
          "   (0 on THESE binary untied maps only --")
    print("                                   NOT a theorem: it fails for C > 2 and for"
          " a binary map with ties at 1/2)")
    print(f"  monotonicity violations       : {100 * np.mean(violations):.1f}% of steps")
    print("  (0% here is an OBSERVATION on this generator, not unimodality: H is not")
    print("   guaranteed unimodal -- 27.3, 4.0, 19.8, 0.0 is a counterexample. And even")
    print("   with 0 violations the sqrt allocation is unjustified, because minimizing")
    print("   the Koksma bound does not minimize the error -- rem:minimaxfallacy)\n")


def main():
    maps = synthetic_maps(N_MAPS)
    exact = [exact_integral(*m) for m in maps]
    print(f"{N_MAPS} maps; exact integral computed over every distinct value\n")
    check_v_shape(maps)

    results = {}

    symmetric = min(
        (mean_error(maps, exact, [t, 1 - t]), float(t))
        for t in GRID
        if t < 0.5
    )
    results["symmetric_2pt_oracle"] = {
        "error": symmetric[0], "nodes": [symmetric[1], 1 - symmetric[1]], "params": 1
    }

    asymmetric = min(
        (mean_error(maps, exact, [a, b]), float(a), float(b))
        for a, b in itertools.combinations(GRID, 2)
    )
    results["asymmetric_2pt_oracle"] = {
        "error": asymmetric[0], "nodes": [asymmetric[1], asymmetric[2]], "params": 2
    }

    weighted = min(
        (mean_error(maps, exact, [a, b], [w, 1 - w]), float(a), float(b), float(w))
        for a, b in itertools.combinations(GRID, 2)
        for w in WEIGHTS
    )
    results["asymmetric_2pt_free_weights_oracle"] = {
        "error": weighted[0],
        "nodes": [weighted[1], weighted[2]],
        "weights": [weighted[3], 1 - weighted[3]],
        "params": 3,
    }

    for count in (2, 4, 8, 16, 32):
        results[f"midpoint_uniform_M{count}"] = {
            "error": mean_error(maps, exact, quadrature_nodes("mid", count)),
            "nodes": quadrature_nodes("mid", count),
            "params": 0,
        }

    # The derived rule: same node budget, allocated between the two monotone halves
    # by sqrt(V_L / V_R). Per-image, but only the ALLOCATION adapts -- the estimand
    # is untouched. The falsifiable prediction is that this beats uniform at equal M.
    ratios = []
    for count in (2, 4, 8, 16, 32):
        errors = []
        allocations = []
        for (_, tensor, predicted), truth in zip(maps, exact):
            nodes, weights, m_left, ratio = variation_allocated_nodes(
                tensor, predicted, count
            )
            risk = quadrature_risks(tensor, predicted, nodes, weights=weights)
            errors.append(abs(risk["hd95"] - truth) / truth * 100)
            allocations.append(m_left)
            if count == 32:
                ratios.append(ratio)
        results[f"variation_allocated_M{count}"] = {
            "error": float(np.mean(errors)),
            "nodes": [],
            "params": 0,
            "mean_left_nodes": float(np.mean(allocations)),
        }

    print(f"{'rule':<42}{'params':>7}{'mean err %':>12}   nodes")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["error"]):
        nodes = ", ".join(f"{x:.3f}" for x in r["nodes"][:6])
        extra = ""
        if "mean_left_nodes" in r:
            extra = f"mean M_L = {r['mean_left_nodes']:.1f}"
        print(f"{name:<42}{r['params']:>7}{r['error']:>11.1f}%   [{nodes}] {extra}")

    print(
        f"\nV_L / V_R over the {len(ratios)} maps: "
        f"median {np.median(ratios):.3f}, "
        f"so theory allocates sqrt of that -> "
        f"{100 * math.sqrt(np.median(ratios)) / (1 + math.sqrt(np.median(ratios))):.0f}%"
        " of the nodes to the LOW-threshold half"
    )
    print("HEAD-TO-HEAD at equal node budget (the falsifiable prediction):")
    for count in (2, 4, 8, 16, 32):
        uniform = results[f"midpoint_uniform_M{count}"]["error"]
        allocated = results[f"variation_allocated_M{count}"]["error"]
        verdict = "allocated WINS" if allocated < uniform else "uniform wins"
        print(
            f"  M={count:<3} uniform {uniform:6.1f}%   allocated {allocated:6.1f}%"
            f"   {verdict}"
        )

    output = REPO_ROOT / "outputs" / "asymmetric_nodes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
