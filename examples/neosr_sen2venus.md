# Training Sentinel‑2 → Venµs super‑resolution with **neosr** + `sen2venus-pytorch-dataset`

This tutorial shows how to train a modern super‑resolution (SR) model with
[neosr](https://github.com/neosr-project/neosr) on the Sen2Venµs dataset exposed by
this package — covering both the **v1** classes (`Sen2VenusSite` / `Sen2Venus`) and the
in‑progress **v2** class (`Sen2VenusV2`).

> Status: **proposal / draft.** The two helper files referenced below
> (`tools/export_neosr.py` and `neosr/data/sen2venus_dataset.py`) are provided inline
> here and can be materialised on request.

---

## 0. The core mismatch (read this first)

neosr is built around a **disk‑based, RGB, 8‑bit** pipeline:

- Its stock `paired` dataset reads image *files* from `dataroot_lq/` and `dataroot_gt/`
  via OpenCV (`imfrombytes`), decodes **BGR→RGB**, and emits `float32` tensors in
  `[0, 1]` with the dict keys neosr expects: `lq`, `gt`, `lq_path`, `gt_path`.
- Most of its losses/metrics (`vgg_perceptual_loss`, `dists_loss`, `fdl_loss`,
  `color = "y"` conversion) assume **3‑channel RGB**.

Sen2Venµs, by contrast, is **in‑memory, multi‑band (4–8 bands), reflectance** data:
`__getitem__` already returns CHW `float32` tensors scaled to `[0, 1]` (no `/255`
needed) as `(lr, hr)` tuples or `{'lr','hr'}` dicts.

That leaves two integration paths:

| Path | Effort | Bands | Radiometry | Use when |
|------|--------|-------|-----------|----------|
| **A — Export to RGB, use stock `paired`** | minutes, zero code in neosr | 3 (true colour) | 8‑bit, lossy | quick baseline / sanity check / leverage neosr's full GAN+perceptual recipe |
| **B — Custom registered dataset** | one small file | all (4 or 8) | full float `[0,1]` | the real goal: multispectral, radiometrically faithful SR |

Both are documented below. **Path B is the recommended target**; Path A is the
fastest way to see neosr running end‑to‑end.

### Scale factors (Sentinel‑2 10/20 m → Venµs 5 m)

| subset | bands | neosr `scale` |
|--------|-------|---------------|
| `rgbnir` | B2,B3,B4,B8 (10 m → 5 m) | **2** |
| `all` | + red‑edge upsampled to 10 m | **2** |
| `rededge` | B5,B6,B7,B8A (20 m → 5 m) | **4** |
| **v2** (any subset) | LR is one harmonised 10 m grid | **2** |

---

## 1. Install

```bash
# in one environment
pip install sen2venus-pytorch-dataset          # this package
# neosr is designed to be run from a clone (it auto-discovers archs/datasets)
git clone https://github.com/neosr-project/neosr.git
cd neosr
pip install -e .            # or: uv sync
```

---

## Path A — quickest: export RGB patches, train stock neosr

### A.1 Export script (`tools/export_neosr.py`)

Walks any Sen2Venµs dataset (v1 *or* v2 — both share the `(lr, hr)` interface) and
writes matching true‑colour PNGs into `lq/` and `gt/`. neosr matches LQ↔GT by filename.

```python
"""Export Sen2Venus patches to neosr's paired (lq/gt) PNG layout."""
import os
import cv2
import fire
import numpy as np
from sen2venus import Sen2Venus, Sen2VenusURLs


def export(out_dir="datasets/sen2venus_rgb", root="datasets", subset="rgbnir",
           sites=None, v2=False):
    if v2:
        from sen2venus.dataset.sen2venus import Sen2VenusV2
        ds = Sen2VenusV2(subset=subset, return_type="tuple")
    else:
        sites = sites or Sen2VenusURLs().get_sites_list()
        ds = Sen2Venus(root, sites, subset=subset, return_type="tuple")

    lq_dir, gt_dir = os.path.join(out_dir, "lq"), os.path.join(out_dir, "gt")
    os.makedirs(lq_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for i in range(len(ds)):
        lr, hr = ds[i]
        # bands [0,1,2] = B2,B3,B4 = Blue,Green,Red == exactly OpenCV's BGR order.
        lr_bgr = (lr[:3].clamp(0, 1).numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)
        hr_bgr = (hr[:3].clamp(0, 1).numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)
        name = f"{i:07d}.png"
        cv2.imwrite(os.path.join(lq_dir, name), lr_bgr)
        cv2.imwrite(os.path.join(gt_dir, name), hr_bgr)
    print(f"Exported {len(ds)} pairs to {out_dir}")


if __name__ == "__main__":
    fire.Fire(export)
```

```bash
python tools/export_neosr.py --subset rgbnir --sites '["SUDOUE-4"]'
```

> ⚠️ Path A discards NIR/red‑edge and quantises reflectance to 8‑bit. It exists to get
> a fast, fully‑supported neosr run (including GAN + perceptual losses) on true colour.

### A.2 Config (`options/train_sen2venus_rgb.toml`)

Start from neosr's `options/train_compact.toml` and change only the data paths + scale:

```toml
name = "sen2venus_rgb_x2"
model_type = "image"
scale = 2
use_amp = true

[datasets.train]
type = "paired"
dataroot_gt = "datasets/sen2venus_rgb/gt/"
dataroot_lq = "datasets/sen2venus_rgb/lq/"
patch_size = 64           # crop from the LQ (GT crop = 64*scale)
batch_size = 16

[network_g]
type = "compact"          # 3-channel default is fine here

[train]
[train.optim_g]
type = "adamw"
lr = 1e-4
[train.mssim_opt]
type = "mssim_loss"
loss_weight = 1.0

[logger]
total_iter = 100000
save_checkpoint_freq = 5000
use_tb_logger = true
```

```bash
python train.py -opt options/train_sen2venus_rgb.toml      # or: neosr-train options/train_sen2venus_rgb.toml
```

---

## Path B — recommended: custom registered dataset (all bands, full radiometry)

neosr auto‑imports every `neosr/data/*_dataset.py` at startup and registers any class
decorated with `@DATASET_REGISTRY.register()` (it is then selectable by its **class
name** via the config `type`). Because our dataset already produces CHW `float32`
tensors in `[0, 1]`, the bridge is tiny — no file export, no `/255`, all bands kept.

### B.1 Bridge dataset (`neosr/data/sen2venus_dataset.py`)

Drop this file into your **neosr clone's** `neosr/data/` folder:

```python
"""neosr ⇆ sen2venus-pytorch-dataset bridge.

Registers a `sen2venus` dataset that yields neosr-style samples
({'lq','gt','lq_path','gt_path'}) directly from the in-memory Sen2Venus tensors,
preserving every band and the native float reflectance range [0, 1].
"""
import torch
from torch.utils import data

from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class sen2venus(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        subset = opt.get("subset", "rgbnir")

        if opt.get("v2", False):
            from sen2venus.dataset.sen2venus import Sen2VenusV2
            self.ds = Sen2VenusV2(
                source=opt.get("source", "tacofoundation:sen2venus"),
                site_names=opt.get("sites"),
                subset=subset,
                include_swir=False,          # keep LR/HR band counts equal (see notes)
                return_type="dict",
                augment=opt.get("augment", False),
                device="cpu",
            )
        else:
            from sen2venus import Sen2Venus, Sen2VenusURLs
            self.ds = Sen2Venus(
                root=opt["dataroot"],
                site_names=opt.get("sites") or Sen2VenusURLs().get_sites_list(),
                subset=subset,
                return_type="dict",
                augment=opt.get("augment", False),
                device="cpu",
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]
        name = f"sen2venus_{idx:07d}"
        return {
            "lq": s["lr"].clamp(0, 1).float(),
            "gt": s["hr"].clamp(0, 1).float(),
            "lq_path": name,
            "gt_path": name,
        }
```

Then make sure the `sen2venus` package is installed in the same env:
`pip install sen2venus-pytorch-dataset`.

### B.2 Config (`options/train_sen2venus_ms.toml`)

A **generator‑only** recipe (no GAN/perceptual, since those assume RGB), training a
4‑channel `compact` ×2 on the `rgbnir` subset. The crucial bits are `num_in_ch` /
`num_out_ch` matching your band count, and a channel‑agnostic loss set.

```toml
name = "sen2venus_ms_x2"
model_type = "image"
scale = 2
use_amp = true

[datasets.train]
type = "sen2venus"             # ← our registered class name
dataroot = "datasets"          # where the .7z are downloaded/extracted (v1)
subset = "rgbnir"              # 4 bands: B2,B3,B4,B8  → set num_*_ch = 4 below
# v2 = true                    # ← switch to the tacoreader v2 dataset instead
# sites = ["SUDOUE-4", "FR-LAM"]
augment = true
batch_size = 16
augmentation = [ "none" ]      # disable mixup/cutblur to start simple
num_worker_per_gpu = 4

[network_g]
type = "compact"
num_in_ch = 4                  # = number of bands in `subset`
num_out_ch = 4
# `upscale` is taken from the top-level `scale`

# NOTE: no [network_d] and no gan_opt → metagan/RGB discriminators assume 3 channels.

[train]
ema = 0.999
[train.optim_g]
type = "adamw"
lr = 1e-4
betas = [ 0.9, 0.99 ]

# channel-agnostic losses only:
[train.mssim_opt]
type = "mssim_loss"
loss_weight = 1.0
[train.ldl_opt]
type = "ldl_loss"
loss_weight = 1.0
# AVOID: vgg_perceptual_loss, dists_loss, fdl_loss (RGB CNNs → crash on N≠3 bands)

[logger]
total_iter = 200000
save_checkpoint_freq = 5000
use_tb_logger = true
```

For `subset = "all"` use `num_in_ch = num_out_ch = 8`; for `rededge` set `scale = 4`
and `num_*_ch = 4`.

```bash
python train.py -opt options/train_sen2venus_ms.toml
```

---

## 3. v2 (`Sen2VenusV2`) specifics

`Sen2VenusV2` is a drop‑in for the same `(lr, hr)` / `{'lr','hr'}` contract, so the
Path‑B bridge already supports it via `v2 = true`. Two things to keep in mind:

- **SWIR asymmetry.** v2 adds B11/B12 — but **only on the Sentinel‑2 (LR) side**; Venµs
  has no SWIR. Leave `include_swir=False` for SR training, otherwise LR has more bands
  than HR and `num_in_ch ≠ num_out_ch` (most SR archs require equal in/out channels).
- **Band order differs from v1.** v2 native order is `[B2,B3,B4,B5,B6,B7,B8,B8A]`
  (red‑edge interleaved), exposed on `ds.lr_bands` / `ds.hr_bands`. If you mix v1‑ and
  v2‑trained checkpoints, account for the reordering.
- v2 streams via `tacoreader` (no Zenodo download); `dataroot` is ignored — set
  `source`/`sites` instead.

---

## 4. Gotchas checklist

- **Channels:** set `num_in_ch`/`num_out_ch` to the band count for whatever arch you
  pick. `compact`, `span`, `esrgan` expose these; many transformer archs hardcode 3 —
  see neosr's *Arch‑specific options* wiki before choosing.
- **Losses/metrics:** anything backed by an ImageNet CNN (`vgg_perceptual_loss`,
  `dists_loss`, `fdl_loss`) or `color = "y"` assumes RGB. Stick to `mssim_loss`,
  `ldl_loss`, pixel `l1`, `fft`/`ff_loss` for >3 bands. neosr's PSNR/SSIM val metrics
  also Y‑convert by default — validate on the RGB subset, or compute per‑band metrics
  yourself.
- **No `/255`:** Sen2Venµs is already `[0, 1]` reflectance. The bridge clamps but does
  not normalise — don't add neosr `mean`/`std`.
- **Patch size:** v1 patches are fixed‑size per subset, so batching works without
  cropping; if you want smaller crops for memory, add a random crop in `__getitem__`.
- **Registration scope:** the auto‑discovery only scans the installed/cloned
  `neosr/data/` folder — the bridge file must live there (or be imported before
  neosr's `build_dataset` runs).

---

## 5. Sources / references
- neosr — <https://github.com/neosr-project/neosr> (Configuration Walkthrough, Losses,
  Arch‑specific options wikis); template configs under `options/train_*.toml`.
- `compact` arch channel params (`num_in_ch`/`num_out_ch`, `upscale` from `scale`):
  `neosr/archs/compact_arch.py`.
- Dataset contract (`lq`/`gt`/`lq_path`/`gt_path`, `[0,1]` float, registry):
  `neosr/data/paired_dataset.py`, `neosr/data/__init__.py`.
