# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-02

### Added

- `Sen2VenusV2` dataset class for the Sen2Venµs **v2.0** release, accessed through the
  [tacofoundation](https://huggingface.co/datasets/tacofoundation/sen2venus) mirror via
  `tacoreader` (no manual Zenodo download). Returns `(lr, hr)` torch tensors with the
  same conventions as `Sen2VenusSite` (`subset`, `return_type`, `device`, `augment`,
  `getitem_xarray`).
- New publicized `include_swir` argument (default `False`) to append the v2 SWIR bands
  (B11, B12) to the Sentinel-2 (LR) tensor. Off by default to match v1 behaviour.
- `Sen2VenusV2.available_sites()` and per-site filtering via `site_names`.
- Optional install extra `pip install sen2venus-pytorch-dataset[v2]` (pins
  `tacoreader<1.0`, the only line that can read the legacy tortilla format).
- Offline `pytest` test suite under `tests/` (no dataset download required) plus a
  GitHub Actions `Tests` workflow running it on Python 3.9–3.12. Installable via the
  `[test]` extra.

### Changed

- Refactored the Kornia augmentation pipeline into a shared `_build_augmentations()`
  helper reused by both v1 and v2.
- Migrated packaging from `setup.py` to a PEP 621 `pyproject.toml` (setuptools
  backend); the version is still sourced from `sen2venus.__version__`.

## [0.0.3] - 2024-02-10

### Added

- progress bar for download
- arguments for train_spl.py example
- README.md in examples/
- retry management during download - with `force_download=True` on second attempts
- add `strict` argument that defaults to `False` to folder validity check, to handle missing GPKG files.

### Changed
- repo renamed to sen2venus-pytorch-dataset after discussion with dataset authors
- removal of non-dataset URL due to bad copy-pasting in list of datasets URLs
 
## [0.0.2] - 2024-01-05

### Added

- New site-specific class `Sen2VenusSite` that incorporates most features from former `Sen2Venus` class
- Helper classes `Sen2VenusSubsetSuffixes` and `Sen2VenusURLs`
- 2 basic tests under examples/
- add multi-resolution loading, via a new `subset` parameter value handling (`'all'`)

### Changed

- `Sen2Venus` class now inherits torch's ConcatDataset class and creates multi-sites concatenated datasets based on a list of `Sen2VenusSite` instances
- example image for README.md is now showcasing 8-band interpolated input
- download process is cleaner, checks for existing compressed files and checks uncompressed folder for required files
- better management of existing dataset folders, removal of the .7z files.