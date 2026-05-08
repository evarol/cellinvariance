# Findings: Cross-Modality Cell Re-Identification (w8)

## Problem

In a paired GCaMP microscopy dataset, the same neurons are imaged twice — once
*in vivo* (`invivo_raw`) and once in fixed *ex-vivo* tissue
(`exvivo_registered`). 115 cells have been hand-annotated in both modalities;
each pair shares a 3D landmark position. The task: given a small image patch
around a cell in one modality, retrieve the same cell in the other modality
using only the learned image representation — no test-time access to landmark
coordinates.

Why it's hard: the two modalities differ in fixation, lighting, optical
sectioning, and registration artefacts. Within a single modality, neighborhood
appearance is consistent enough that frozen DINO features cluster cells very
cleanly (within-modality knn accuracy ≈ 99.8%). Across modalities, the same
cell can look quite different.

## Dataset

- 115 paired landmark cells.
- Two volumes: `zstack.tif` (in vivo) and `Sparrow_3_po_488_4x-registered.tif`
  (ex vivo, registered to in-vivo coordinates).
- 50–80 µm xy patches × 12–36 µm z extracted around each landmark.
- Loader and configuration come from
  `cellfind/configs/rotation_domain_invariance.yaml` (read-only source).
- Evaluation is leave-one-out cross-validation over the 115 cells.

## Method

w8 = `w8_e200_ms3_hi_ap800_j0_xy80z24_skh07_dv2_g0`. The pipeline:

1. **Frozen feature extractor**: `facebook/dinov2-small`. CLS embeddings are
   pulled at three spatial scales — `(40, 80, 120) µm` xy with matching
   `(12, 24, 36) µm` z — and concatenated into a 1152-d feature.
2. **Augmentation**: `hard_intra` preset (rotation + light photometric
   perturbations), 800 augmentations per cell. The frozen DINO features for
   all augmentations are precomputed once and cached
   (`.feature_cache/emb_cache_w8/`).
3. **MLP projector**: `1152 → 256 (ReLU) → 64 (L2-normalized)`. This is the
   only trained component.
4. **Loss**:
   - Supervised contrastive (intra-modality): same cell across augmentations
     is pulled together; different cells are pushed apart. Temperature 0.1.
   - Sinkhorn cross-modality alignment (λ = 0.7): aligns the empirical
     distribution of in-vivo embeddings to ex-vivo embeddings, *without*
     using paired cell labels across modalities.
5. **Training**: 200 epochs, batch 1024, AdamW, lr 1e-3, weight decay 1e-4.
6. **Evaluation**: per cell, hold it out, encode its ex-vivo patch, retrieve
   the nearest in-vivo patches by cosine similarity. Report R@1, R@5
   (cell-prototype = mean of patch embeddings per cell), and MRR.

## Result

From [`w8_results.json`](w8_results.json):

| Metric | Value |
|---|---|
| `ex_to_iv_top1_acc` (R@1) | **0.0887** |
| `ex_to_iv_top5_acc_cell_proto` (R@5) | **0.2498** |
| `ex_to_iv_top5_acc_patch_knn` | 0.1055 |
| `ex_to_iv_mrr_cell_proto` | 0.1742 |
| `best_balanced_val_knn_acc` | 0.9979 |
| `best_iv_val_knn_acc` | 0.9971 |
| `best_ex_val_knn_acc` | 0.9986 |

The within-modality balanced knn accuracy of 99.8% says the projector keeps
cells of the same modality almost perfectly clustered. The R@1 of 8.9% on the
cross-modality task is, by contrast, modest — it tells us the bottleneck is
the in-vivo ↔ ex-vivo *bridge*, not the per-modality representation. The R@5
of 25% means that for one in four cells, the correct match is in the top five
candidates returned by the system.

## What this means

- DINOv2 features on small GCaMP patches carry enough signal to solve
  *within-modality* re-identification almost perfectly. The model isn't blind
  to cell identity.
- Sinkhorn alignment on its own — without paired cross-modality labels —
  closes only a small part of the modality gap on this dataset. The fixation
  / illumination shift between in-vivo and ex-vivo is large relative to the
  within-modality variation.
- Within the constraint of "appearance only, no test-time spatial priors",
  this is the best of the wave-numbered method iterations on a DINO backbone.

## Reproduce

```bash
# regenerate the visualization from the bundled w8 projector (~5–15 min, runs DINO once)
uv sync
uv run python scripts/reproduce_w8.py --skip-train

# full retrain from scratch (~12–20 h on MPS for the first run; ~30 min after the embedding cache exists)
uv run python scripts/reproduce_w8.py
```

The first full run spends most of its time computing frozen DINOv2 features
for the ~92 000 augmented patches per modality and writing them to
`.feature_cache/emb_cache_w8/`. Subsequent runs reuse that cache.

The interactive visualization
[`viz/dino_repr_interactive.html`](../viz/dino_repr_interactive.html) shows
three things side-by-side:

1. PCA-2D scatter of the cells in three representation spaces — the raw
   1152-d DINO concat, the 256-d MLP middle layer, and the 64-d MLP output.
   Both modalities are overlaid with augmentation dots clustered around each
   base patch. Watch how cells separate from the input space to the output.
2. The full 115×115 cosine-similarity matrix for the trained 64-d
   representation.
3. A contact sheet: for every ex-vivo query, the top-5 in-vivo retrievals,
   with the correct match highlighted when it falls in the top-5.

## Limitations

- Single backbone (DINOv2-small). The wave that swapped in DINOv3-ViT-L,
  Phikon, Phikon2, CTransPath, etc., did not improve appearance-only R@1 on
  this dataset; reported for context but not included here.
- Single dataset, 115 cells. No bootstrap or cross-validation across
  landmark splits beyond LOOCV.
- The original wave runner reports `top1_acc`, `top5_acc_cell_proto`,
  `top5_acc_patch_knn`, and a `topk_purity` cluster-level metric; these are
  what the visualization and findings reference. R@10 is not measured.
