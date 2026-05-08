# Cell Invariance — Cross-Modality Cell Re-Identification

Re-identify the same neuron across two GCaMP microscopy modalities — *in
vivo* (`invivo_raw`) and fixed *ex vivo* (`exvivo_registered`) — from a small
image patch around the cell. 115 paired landmark cells, frozen DINOv2-small
features, supervised-contrastive + Sinkhorn cross-modality alignment, no
test-time access to landmark coordinates.

## Headline

| Metric | Value |
|---|---|
| LOOCV R@1 (ex → iv) | **0.0887** |
| LOOCV R@5, cell-prototype | **0.2498** |
| MRR, cell-prototype | 0.1742 |
| Within-modality balanced knn accuracy | 0.9979 |

The within-modality features are nearly perfect; the cross-modality bridge is
the hard part.

## Method (w8)

1. Frozen `facebook/dinov2-small`, multi-scale CLS at xy ∈ {40, 80, 120} µm,
   z ∈ {12, 24, 36} µm → concatenated 1152-d feature.
2. 800 `hard_intra` (rotation-heavy) augmentations per cell, embedded once
   and cached.
3. MLP projector 1152 → 256 → 64 (L2-normalized), trained 200 epochs with
   supervised contrastive (τ = 0.1) + Sinkhorn cross-modality alignment
   (λ = 0.7). Batch 1024, AdamW, lr 1e-3.
4. Evaluation: per-cell leave-one-out, ex → iv top-K cosine retrieval.

See [`results/findings.md`](results/findings.md) for the full write-up.

## Quick start

```bash
uv sync
uv run python scripts/reproduce_w8.py --skip-train  # regenerate viz from bundled projector (~10 min)
```

The bundled [`viz/dino_repr_interactive.html`](viz/dino_repr_interactive.html)
already shows the trained model. The line above re-derives it from
[`results/w8_projector.pt`](results/w8_projector.pt) by computing fresh DINO
features.

Full retrain from scratch (first run takes 12–20 h on MPS to embed all
augmented patches; subsequent runs are ~30 min):

```bash
uv run python scripts/reproduce_w8.py
```

## Visualization

`viz/dino_repr_interactive.html` is a self-contained interactive HTML with
three coupled panels:

1. **Representation scatter** — PCA-2D of the 115 cells in the raw 1152-d
   DINO space, the 256-d MLP middle layer, and the 64-d MLP output. Both
   modalities (iv ●, ex ◆) overlaid; same color per cell. Augmentations as
   small translucent dots.
2. **Discriminability matrix** — 115×115 cosine similarity over the trained
   64-d representation.
3. **EX → IV contact sheet** — for every ex-vivo query, the top-5 in-vivo
   retrievals as image thumbnails, with the correct match highlighted when
   in the top-5.

## Repo layout

```
.
├── src/                 # core library (frozen DINO pipeline + supervised contrastive + sinkhorn)
│   ├── runner.py        # patch extraction, multi-scale DINOv2 feature pipeline
│   ├── contrastive.py   # MLPProjector + LOOCV training & evaluation
│   ├── augmentations.py # rotation augmentation presets
│   └── alignment.py     # Sinkhorn cross-modality alignment loss
├── scripts/
│   └── reproduce_w8.py  # end-to-end orchestrator (train → install → regen viz)
├── viz/
│   ├── generate_data.py             # build the data JSON from a trained projector
│   ├── generate_html.py             # render the data JSON to an interactive HTML
│   └── dino_repr_interactive.html   # bundled visualization (15 MB)
├── results/
│   ├── w8_projector.pt    # bundled trained MLP projector weights (1.2 MB)
│   ├── w8_results.json    # full LOOCV metrics
│   ├── w8_folds.jsonl     # per-fold breakdown
│   ├── w8_training.log    # original training log
│   └── findings.md        # detailed write-up
├── pyproject.toml
└── README.md
```

## Data dependency

The training and visualization scripts read raw image volumes from a
read-only `cellfind` checkout:

- `cellfind/configs/rotation_domain_invariance.yaml` (dataset config)
- `cellfind/datasets/zstack.tif` (in vivo)
- `cellfind/datasets/Sparrow_3_po_488_4x-registered.tif` (ex vivo)
- `cellfind/datasets/slice3_to_invivoLANDMARKS.json` (115 paired landmarks)

By default the code looks in `/Users/erdem/Documents/github/cellfind`. Set
`CELLFIND_ROOT=/path/to/cellfind` to point elsewhere.

These files are *not* committed in this repo.

## License

MIT — see [`LICENSE`](LICENSE).
