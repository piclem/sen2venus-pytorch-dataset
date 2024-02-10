# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
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