# Third-party code — PLAS (Point Label Aware Superpixels)

This directory is derived from **Point Label Aware Superpixels** by Scarlett Raine et al.:

    upstream:  https://github.com/sgraine/point-label-aware-superpixels
    commit:    6049fe4a8eaf5380fff8e46df4cb881f6f11e607  (2022-09-15)
    licence:   GNU General Public License v3.0 — see LICENSE in this directory

GPL-3.0 applies to everything in this directory. The AGPL-3.0 in the repository root
covers the rest of SSeg; the two are compatible, and this code keeps its own licence.

If you use this component, cite the original work:

```bibtex
@ARTICLE{9813385,
  author={Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Fischer, Tobias},
  journal={IEEE Robotics and Automation Letters},
  title={Point Label Aware Superpixels for Multi-Species Segmentation of Underwater Imagery},
  year={2022}, volume={7}, number={3}
}
```

## Why it is vendored rather than declared as a dependency

Upstream ships as a set of scripts — no `setup.py`, no `pyproject.toml` — so there is
nothing to `pip install`, and the files here carry modifications besides. Unlike SAM 2,
which this repo now takes as a pinned dependency, PLAS has to travel with the code.

## The checkpoint

`checkpoints/download_ckpts.sh` also fetches `standardization_C=100_step70000.pth`, the
pretrained SSN feature extractor these superpixels rely on. That weight file is the PLAS
authors', distributed from their Google Drive, and is covered by the same licence.

## What is here, and what we changed

Modified by the SSeg authors, 2025–2026. Per file:

**`ssn.py`** — unmodified. Itself credits the PyTorch SSN implementation at
https://github.com/andrewsonga/ssn_pytorch and the original SSN paper.

**`spixel_utils.py`** — two changes:
- `get_spixel_init`: the `torch.meshgrid` call used the deprecated positional-list form
  with `out=torch.FloatTensor()`, which pins the grid to CPU. It now builds on the input
  tensor's device and passes `indexing="ij"`.
- `compute_init_spixel_feature`: dropped the `torch_scatter` dependency. The mean feature
  per superpixel is now computed with `index_add_` plus a count, guarding empty
  superpixels, instead of `torch_scatter.scatter(..., reduce='mean')`.

**`segmenter_plas.py`** — upstream's `propagate.py`, restructured into a
`SuperpixelLabelExpander` class so SSeg can call it per image instead of running a script.
Five of upstream's six top-level definitions live inside `expand_labels`:
- `enforce_connectivity` — verbatim.
- `members_from_clusters` — verbatim apart from `device` becoming `self.device`.
- `CustomLoss`, `optimize_spix`, `prop_to_unlabelled_spix_feat` — modified, most heavily
  `prop_to_unlabelled_spix_feat`.
- `plot_propagated` — dropped; SSeg renders its own overlays.

Added by SSeg, not present upstream: `generate_segmented_image`, and seeding /
determinism control in `__init__`.

To reproduce this comparison:

    git clone https://github.com/sgraine/point-label-aware-superpixels
    cd point-label-aware-superpixels && git checkout 6049fe4
    diff propagate.py     <sseg>/plas/segmenter_plas.py
    diff spixel_utils.py  <sseg>/plas/spixel_utils.py
