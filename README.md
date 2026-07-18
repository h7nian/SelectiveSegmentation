# Selective Segmentation

Research code for **loss-indexed confidence in selective binary
segmentation**.  A foreground probability map `p` and the deployed mask
`Y_hat = {p >= gamma}` induce the midpoint confidence

```text
C[L, M] = -(1 / M) sum_m L({p >= (m - 1/2) / M}, Y_hat).
```

Changing `L` aligns confidence with a target risk; changing `M` changes only
the numerical approximation.  The focused benchmark uses foreground Dice
loss and image-diagonal-normalized penalized HD95 (nHD95), and evaluates the
resulting rankings with tie-aware AURC.  The manuscript is in
[`docs/main.tex`](docs/main.tex).

## Install

Python 3.10 or newer is required.  An editable install exposes the module
CLIs and three equivalent console entry points.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,plots]"
```

`requirements.txt` records the versions used in the current environment for
an exact local reproduction.  On the MSI cluster, load the Python module and
source `scripts/slurm/env.sh` before running jobs.

## Data

The primary experiment uses one native binary task per image.  It never
selects a class from the ground truth.

| key | target | split policy | expected directory |
| --- | --- | --- | --- |
| `pet` | pet including trimap border | official `trainval` / `test` | `data/oxford-iiit-pet` |
| `kvasir` | polyp | deterministic SHA-256-ranked 80/20 split | `data/Kvasir-SEG/{images,masks}` |
| `fives` | retinal vessels | official `train` / `test` | `data/FIVES/{train,test}/{Original,Ground truth}` |

Masks are validated as total binary labels at evaluation time. Pet's border
is foreground by a predeclared policy. Kvasir-SEG thresholds the maximum JPEG
mask channel at 128 to reject compression residuals; FIVES maps every nonzero
mask value to foreground. Dataset archives, caches, and checkpoints are local
artifacts and are excluded from Git.

Download datasets and populate the repo-local model caches with:

```bash
python scripts/download_binary_assets.py
```

FIVES extraction additionally requires `unrar`; every downloaded archive is
checked against the SHA-256 recorded in the script.

## Reproduce the focused workflow

Fine-tune a target model:

```bash
python -m selectseg.train \
  --model clipseg --dataset kvasir \
  --output-dir outputs/train/clipseg_kvasir
```

Evaluate confidence and per-image risks.  Omitting `--checkpoint` gives the
pretrained CLIPSeg condition; passing it gives the target-adapted condition.
The default action threshold is `gamma=0.5`, and the default midpoint ladder
is `M in {2, 8, 32}`.

```bash
python -m selectseg.binary_eval \
  --model clipseg --dataset kvasir \
  --checkpoint outputs/train/clipseg_kvasir/checkpoint.pt \
  --decision-threshold 0.5 --m-values 2 8 32
```

Each run writes a content-addressed directory under
`outputs/binary/<dataset>/<condition>/<run-id>/` containing strict JSONL rows
and a provenance manifest.  Analyze all completed runs with:

```bash
python scripts/analyze_binary.py \
  --input-dir outputs/binary \
  --output-dir outputs/binary/analysis
```

The analysis emits JSON, long-form CSV, and a LaTeX table.  It reports raw,
excess, and normalized AURC; paired image-level percentile-bootstrap intervals
and two-sided bootstrap tail p-values; and Holm-adjusted p-values. Exact score
ties are averaged analytically rather than broken by input order.

The released `results/analysis.json` is the canonical machine-readable final
analysis, and `results/main_table.csv` is its long-form projection. Generated
paper tables record the analysis JSON SHA-256 in their source comments. The ten
`results/manifests/` files retain input, checkpoint, source, environment, and
sample-list hashes. Raw datasets, checkpoints, caches, and per-image JSONL files
remain outside Git.

DeepLabV3 is supported for target fine-tuning.  Its external COCO checkpoint
is only meaningful when the dataset vocabulary maps to checkpoint classes
(Pet in the binary benchmark); it is not a zero-shot baseline for polyps or
retinal vessels.

## Verify

```bash
pytest -q tests/test_binary_framework.py \
  tests/test_binary_eval.py tests/test_analyze_binary.py tests/test_data.py
ruff check .
```

The focused modules have no dependency on the legacy selective pipeline.  A
quick dependency check is:

```bash
rg "selectseg\.selective" selectseg/binary_framework.py \
  selectseg/binary_eval.py scripts/analyze_binary.py
```

No matches are expected.

## Repository map

```text
selectseg/binary_framework.py  losses, confidence, tie-aware AURC, bootstrap
selectseg/binary_eval.py       strict one-row-per-image binary evaluator
selectseg/data.py              dataset specifications, validation, transforms
selectseg/models.py            CLIPSeg and DeepLabV3 adapters
selectseg/train.py             target fine-tuning CLI
scripts/analyze_binary.py      strict statistical analysis and table export
tests/                         unit, schema, data, and pipeline tests
docs/                          ICLR manuscript source
```

The earlier multiclass/band implementation remains only in the private
development workspace for provenance.  It is intentionally omitted from the
focused public mirror and is not part of this workflow or its claims.
