# Known-posterior synthetic protocol v1

This auxiliary campaign isolates the error caused by the shared-threshold
working posterior. It never changes the main binary campaign and never writes
sampled masks.

## Frozen design

- True posterior couplings: shared threshold (well specified), conditionally
  independent Bernoulli, local 8-by-8 block thresholds, and a two-mode
  antithetic/non-nested coupling.
- Marginal sharpness: diffuse, medium, and sharp.
- Foreground morphology: disk, elongated, and two-component.
- Replicates: 10 deterministic replicate identities.
- Total: `4 x 3 x 3 x 10 = 360` independent cells.
- Each cell contains 24 probability maps and 512 posterior draws per image,
  split into eight Monte Carlo batches. Four CPU workers process images in
  parallel without changing the deterministic result.
- The deployed action is fixed at `p >= 0.5`. The working posterior is always
  induced by one shared `Uniform(0,1)` threshold.

For each loss (Dice, normalized penalized HD, and normalized penalized HD95),
the cell records true-P risk, Q risk, score/rank error, tie-aware AURC regret,
and posterior-integration Monte Carlo SE. It evaluates midpoint quadrature at
`M = 2, 8, 32, 128`; only Dice additionally has an exact probability-knot
integral.

Posterior diagnostics include exact TV for these constructed finite laws, an
exact empty-event TV lower bound, paired Jaccard/full-HD transport-cost upper
bounds, and empirical scalar loss-pushforward W1. The HD95 pushforward
diagnostic is not an HD95 mask-Wasserstein theorem.

## Pilot and gate

The pilot is the disk, replicate-0 slice: four couplings by three sharpness
levels, exactly 12 jobs. The remaining full plan contains exactly 348 jobs.
Expansion requires:

1. shared-threshold P and Q paired Monte Carlo risks and discrepancy diagnostics
   to agree exactly;
2. shared-threshold Dice-Exact mean error to lie within three reported Monte
   Carlo SEs; and
3. every pilot runtime to remain below 10,800 seconds, retaining 25% headroom
   under the four-hour allocation.

## Commands (dry-run by default)

```bash
python -m scripts.submit.synthetic --phase pilot
python -m scripts.submit.synthetic --phase pilot --submit \
  --receipt outputs/synthetic_posterior_campaign/pilot-submissions.jsonl
```

After the 12 pilot manifests exist, aggregate them and record the analysis
SHA-256:

```bash
python -m scripts.analyze.synthetic \
  --lock configs/auxiliary/synthetic_posterior-v1.lock.json \
  --mode pilot \
  --output outputs/synthetic_posterior_analysis/pilot-analysis.json
sha256sum outputs/synthetic_posterior_analysis/pilot-analysis.json
python -m scripts.render.synthetic \
  --analysis outputs/synthetic_posterior_analysis/pilot-analysis.json \
  --output-dir outputs/synthetic_posterior_analysis/pilot-render
```

Preview and submit the remaining 348 cells only by supplying that analysis and
its explicit digest. The submitter strictly reloads the 12 locked pilot cells
and recomputes the gate before it creates either plan. Full submission has one
fixed append-only duplicate guard:

```bash
python -m scripts.submit.synthetic --phase full \
  --pilot-analysis outputs/synthetic_posterior_analysis/pilot-analysis.json \
  --expected-pilot-analysis-sha256 <pilot-analysis-sha256>
python -m scripts.submit.synthetic --phase full \
  --pilot-analysis outputs/synthetic_posterior_analysis/pilot-analysis.json \
  --expected-pilot-analysis-sha256 <pilot-analysis-sha256> --submit \
  --receipt outputs/synthetic_posterior_campaign/full-submissions.jsonl
```

Complete analysis requires the 12 pilot artifacts plus the 348 non-pilot
artifacts and uses `--mode complete`. The submitter uses no arrays.
