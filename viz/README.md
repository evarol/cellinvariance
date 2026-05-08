# `viz/` — interactive visualization

`dino_repr_interactive.html` is a self-contained HTML file rendered by
[`generate_html.py`](generate_html.py) from a JSON produced by
[`generate_data.py`](generate_data.py).

It has three panels, all sharing the same per-cell color palette:

1. **Three-panel representation scatter** — PCA-2D of the 115 cells in three
   spaces side-by-side. Left: 1152-d concatenated DINOv2-small CLS over three
   spatial scales (the input to the trained projector). Middle: the 256-d
   ReLU mid-layer of the MLP. Right: the final 64-d L2-normalized output.
   Both modalities (in vivo ●, ex vivo ◆) are overlaid, with 12 augmentation
   embeddings per cell shown as small translucent dots clustered around each
   base point. Use the legend to toggle modalities and augmentations.
2. **Discriminability matrix** — full 115×115 cosine-similarity heat map over
   the trained 64-d representation. The diagonal is the within-cell
   ex-vivo→in-vivo similarity (the signal we want to be high); off-diagonal
   is between-cell similarity (we want low).
3. **EX → IV top-5 contact sheet** — every ex-vivo query cell shown with its
   five best-matching in-vivo retrievals as thumbnail images. The correct
   match (if it appears in the top-5) is highlighted.

## Regenerate

```bash
uv run python viz/generate_data.py    # ~5–10 min: runs frozen DINOv2 once
uv run python viz/generate_html.py    # ~few seconds
```

`generate_data.py` reads:

- `../results/w8_projector.pt` — the trained MLP projector
- `../results/w8_results.json` — for the metrics displayed in the HTML header
- The cellfind dataset (see top-level `README.md`)

It writes `dino_repr_data.json` next to itself, which `generate_html.py`
then renders to `dino_repr_interactive.html`.

`scripts/reproduce_w8.py --skip-train` runs both in sequence.
