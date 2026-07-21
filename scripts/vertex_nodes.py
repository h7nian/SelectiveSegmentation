"""Where is H's vertex, and do nodes that straddle it beat nodes that do not?

H(t) = L(Y_t, Yhat_c) is EMPIRICALLY V-shaped (obs:vshape -- an observation, not a
theorem): it falls as the level set approaches the prediction and rises as it moves
away again.

TWO CLAIMS THIS SCRIPT USED TO MAKE ARE REFUTED, AND IT USED TO *MEASURE* THE SECOND.

  1. "The vertex sits at t = 1/2, with H(1/2) = 0 because Y_{1/2} IS Yhat." True only
     for a BINARY map, and even there only if NO pixel ties at exactly 1/2 (argmax
     breaks ties toward background, so Yhat = {p > 1/2} STRICTLY while
     Y_{1/2} = {p >= 1/2} keeps the ties).
  2. "The correction is exact: the vertex is the ARGMAX FLOOR phi -- 1/C for softmax,
     1/2 for CLIPSeg and binary." ALSO FALSE, and this is the one that mattered,
     because ``argmax_floor`` fed phi straight into the rule the script published as
     "VERTEX-AWARE (derived)". phi is only a LOWER BOUND on m_c (prop:floor(ii)); it is
     not the vertex, and THERE IS NO CLOSED FORM for the vertex (rem:vshapefails). The
     script's own table had the refutation in it and rationalized it away -- t* sat
     consistently ABOVE phi (0.060 vs 0.048; 0.566 vs 0.500), which we explained as
     "trimming higher matches better" instead of reading it as the floor simply not
     being the vertex.

WHAT SURVIVES (prop:floor), and what this script now measures:

    m_c := min{p_ci : i in Yhat_c}, the smallest probability the map attains inside its
    own prediction. Y_t contains Yhat_c  <=>  t <= m_c, so Y_{m_c} is the TIGHTEST
    level set containing the prediction; and m_c >= phi, the argmax floor. m_c is the
    best cheap ESTIMATE of the vertex, and it is an estimate.

    construction          floor phi   m_c     measured t*   better predictor
    binary                0.500       0.500   0.500         (tie, exact)
    softmax C=5           0.200       0.206   0.223         m_c
    softmax C=21 (VOC)    0.048       0.055   0.060         m_c
    CLIPSeg-style C=21    0.500       0.533   0.550         m_c (3x closer)

m_c is PER MAP, phi is per construction: that is the whole difference, and it is why
the rule below now recomputes t_hat on every map, exactly as eq:derivednodes prescribes
(t_hat := m_c) and exactly as the deployed vtx2 rule in selectseg/selective.py does.

So we test, against the exact gap-weighted integral:

  * the naive binary-derived midpoint (0.25, 0.75)
  * the deployed band's nodes (0.1, 0.9)
  * VERTEX-AWARE nodes (eq:derivednodes): split at t_hat = m_c, midpoint of each half,
    weights equal to the subinterval widths -- so the estimand is untouched and only
    the node placement adapts. This reduces to (0.25, 0.75) exactly when m_c = 1/2.
  * an ORACLE 2-point rule, grid-searched on the very maps it is scored on

across binary, softmax at several C, and a CLIPSeg-style sigmoid map. phi, m_c and the
measured t* are reported as THREE SEPARATE columns and must never be conflated again.

NODE PLACEMENT REMAINS AN OPEN EMPIRICAL QUESTION. Even if the vertex-aware rule wins
here, that settles nothing: minimizing the Koksma bound does not minimize the error
(rem:minimaxfallacy -- a node set with a WORSE star-discrepancy achieves a BETTER
error, and the vertex-aware nodes are themselves an instance), and two synthetic
generators disagree about the optimum. Only the AURC on the real conditions decides.

Run under SLURM (scripts/slurm/vertex.sbatch). CPU-only.
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.selective import quadrature_risks  # noqa: E402

N_MAPS = 24
GRID = np.round(np.arange(0.02, 1.0, 0.02), 3)


def argmax_floor(kind, classes):
    """phi: the smallest probability with which a class CAN win the argmax.

    A softmax winner is the largest of C numbers summing to 1, so it is at least 1/C. A
    CLIPSeg foreground class must beat a background of 1 - max(fg), so it must exceed
    1/2. Binary is the same statement with C = 2.

    THIS IS NOT THE VERTEX OF H. prop:floor(ii) gives only m_c >= phi -- phi is a LOWER
    BOUND on the smallest probability the map ACTUALLY attains inside its prediction,
    and a given map's winner need never descend to it. An earlier version of this
    docstring said "this is the vertex of H, derived rather than searched", and the
    script then built its headline rule on it; that is REFUTED (rem:vshapefails: on a
    3-way softmax whose winner bottoms out at 0.40, H(1/3) = 5.66 while H(0.35) = 0).
    Reported here as a lower bound only, alongside m_c and the measured t*.
    """
    return 1 / classes if kind == "softmax" else 0.5


def vertex_estimate(prob, mask):
    """t_hat := m_c, the min in-mask probability. The rule's actual split point.

    The tightest threshold whose level set still contains the prediction
    (prop:floor(i): Y_t contains Yhat_c iff t <= m_c). An ESTIMATE of H's vertex, not
    the vertex -- there is no closed form (rem:vshapefails) -- but the better of the
    two cheap estimates by a factor of 3 on a CLIPSeg-style map, and it is what
    eq:derivednodes and the deployed vtx2 score both use.
    """
    return float(prob[mask].min())


def vertex_nodes(t_star, count=2):
    """Composite midpoint on the two monotone halves either side of the vertex.

    The weights are the subinterval widths, so the rule still estimates
    int_0^1 H(t) dt exactly -- only the node PLACEMENT adapts. That is what keeps
    this safe where rank_anchored_threshold and importance_nodes were not: they moved
    the estimand, this moves only the quadrature's accuracy.
    """
    left = count // 2
    right = count - left
    nodes = [(j + 0.5) / left * t_star for j in range(left)]
    nodes += [t_star + (j + 0.5) / right * (1 - t_star) for j in range(right)]
    weights = [t_star / left] * left + [(1 - t_star) / right] * right
    return nodes, weights


def make_maps(kind, classes, seed):
    """Spatially coherent probability maps of the requested construction."""
    generator = torch.Generator().manual_seed(seed)
    maps = []
    while len(maps) < N_MAPS:
        logits = torch.randn(classes, 56, 56, generator=generator) * 1.3
        logits = F.avg_pool2d(logits[None], 7, 1, 3)[0]
        if kind == "softmax":
            probs = torch.softmax(logits, dim=0)
        else:  # CLIPSeg: independent sigmoids, background = 1 - max(foreground)
            foreground = torch.sigmoid(logits[1:])
            background = 1 - foreground.max(dim=0, keepdim=True).values
            probs = torch.cat([background, foreground], dim=0)
        prediction = probs.argmax(dim=0)
        for c in range(1, classes):
            mask = prediction == c
            if mask.sum() >= 60:
                maps.append((probs[c], mask))
                break
        if len(maps) and maps[-1][1].sum() < 60:
            maps.pop()
    return maps


def exact_integral(prob, mask):
    """The gap-weighted sum over every distinct probability value: no quadrature error."""
    values = np.unique(prob.numpy())
    edges = np.concatenate([[0.0], values, [1.0]])
    total = 0.0
    for index, value in enumerate(values):
        risk = quadrature_risks(prob, mask, [float(value)])
        height = risk["hd95"] if risk is not None else 0.0
        total += (edges[index + 1] - edges[index]) * height
    return total + (1 - values[-1]) * math.hypot(*prob.numpy().shape)


def error(maps, exact, nodes, weights=None):
    """Mean relative error of a rule against the exact integral.

    quadrature_risks returns None when even the LOWEST node's level set is empty --
    i.e. when every node sits above the map's largest probability. That is not an
    undefined case, it is a fully determined one: by the bounded convention
    L(empty, Yhat) = diag, so every node's loss is the diagonal and the weighted sum
    is the diagonal. Skipping such rules instead would quietly exclude exactly the
    node pairs that are too high for a given map, biasing the ORACLE in its own
    favour -- so we score them at the value the convention assigns.
    """
    errors = []
    for (prob, mask), truth in zip(maps, exact):
        risk = quadrature_risks(prob, mask, list(nodes), weights=weights)
        value = (
            risk["hd95"]
            if risk is not None
            else math.hypot(*prob.numpy().shape)
        )
        errors.append(abs(value - truth) / truth * 100)
    return float(np.mean(errors))


def vertex_aware_error(maps, exact):
    """eq:derivednodes, with t_hat = m_c recomputed PER MAP. Also returns mean m_c.

    This is the rule the paper actually prescribes and the one selectseg/selective.py
    deploys as ``vtx2``. It is NOT the argmax-floor rule this script used to measure:
    m_c is a property of the individual map, phi only of the construction. The
    degenerate guard mirrors vtx2's -- an m_c pinned against 0 or 1 leaves one half of
    the split with zero width, so fall back to the midpoint rule rather than emit a
    degenerate one.
    """
    errors, m_cs = [], []
    for (prob, mask), truth in zip(maps, exact):
        m_c = vertex_estimate(prob, mask)
        m_cs.append(m_c)
        if not 1e-3 < m_c < 1 - 1e-3:
            nodes, weights = [0.25, 0.75], [0.5, 0.5]
        else:
            nodes, weights = vertex_nodes(m_c, 2)
        errors.append(error([(prob, mask)], [truth], nodes, weights))
    return float(np.mean(errors)), float(np.mean(m_cs))


def main():
    settings = [
        ("binary", "softmax", 2),
        ("softmax C=5", "softmax", 5),
        ("softmax C=21 (VOC/DeepLabV3)", "softmax", 21),
        ("CLIPSeg-style C=21", "clipseg", 21),
    ]
    report = {}
    for label, kind, classes in settings:
        maps = make_maps(kind, classes, seed=7 + classes)
        exact = [exact_integral(*m) for m in maps]
        # phi: a LOWER BOUND on m_c, reported but NOT used as the split point.
        phi = argmax_floor(kind, classes)

        # where the vertex actually is, measured
        measured = []
        for prob, mask in maps:
            heights = [quadrature_risks(prob, mask, [float(t)]) for t in GRID]
            heights = np.array(
                [h["hd95"] if h is not None else np.nan for h in heights]
            )
            measured.append(GRID[int(np.nanargmin(heights))])
        measured = float(np.mean(measured))

        # the rule the paper prescribes: t_hat = m_c, recomputed per map
        vertex_error, mean_m_c = vertex_aware_error(maps, exact)
        # the REFUTED rule this script used to publish as "VERTEX-AWARE (derived)":
        # split at the argmax floor. Kept as a named ablation so the refutation is
        # visible in the output rather than merely asserted in a docstring.
        floor_nodes, floor_weights = vertex_nodes(phi, 2)

        # The oracle must search WEIGHTS as well as nodes. The vertex-aware rule
        # carries non-equal weights by construction (the subinterval widths t*, 1-t*),
        # so an equal-weight-only oracle is not an upper bound on it -- and the
        # vertex rule can then "beat the oracle", which is a bug in the comparison,
        # not a result. An equal-weight oracle also degenerates: it will happily put
        # one node AT the vertex, where H = 0, purely to halve its estimate, which is
        # an artefact of the constraint rather than a sensible node.
        oracle = min(
            (error(maps, exact, [a, b], [w, 1 - w]), float(a), float(b), float(w))
            for a, b in itertools.combinations(GRID[::2], 2)
            for w in (0.1, 0.25, 0.5, 0.75, 0.9)
        )
        rules = {
            "naive midpoint (0.25, 0.75)": error(maps, exact, [0.25, 0.75]),
            "deployed band (0.10, 0.90)": error(maps, exact, [0.10, 0.90]),
            "VERTEX-AWARE, t_hat = m_c": vertex_error,
            "REFUTED: split at floor phi": error(
                maps, exact, floor_nodes, floor_weights
            ),
            "oracle 2-point (nodes+weights)": oracle[0],
        }
        report[label] = {
            "argmax_floor_phi": phi,          # a LOWER BOUND on m_c, not the vertex
            "m_c_mean": mean_m_c,             # the vertex ESTIMATE the rule splits at
            "t_star_measured": measured,      # where the minimum actually sits
            "vertex_nodes_at_mean_m_c": list(vertex_nodes(mean_m_c, 2)[0]),
            "oracle_nodes": [oracle[1], oracle[2]],
            "errors": rules,
        }

        print(f"\n=== {label} ===")
        print(
            f"  floor phi = {phi:.3f} (a LOWER BOUND)   m_c = {mean_m_c:.3f} (the "
            f"estimate)   measured t* = {measured:.3f}"
        )
        print(
            "  phi is NOT the vertex (rem:vshapefails); the rule splits at m_c, "
            "per map."
        )
        print(f"  {'rule':<32}{'mean err %':>12}")
        for name, err in sorted(rules.items(), key=lambda kv: kv[1]):
            star = "  <-- eq:derivednodes" if "VERTEX-AWARE" in name else ""
            print(f"  {name:<32}{err:>11.1f}%{star}")
        print(f"  oracle's own nodes: {oracle[1]:.2f}, {oracle[2]:.2f}")

    output = REPO_ROOT / "outputs" / "vertex_nodes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
