# sen2venus
Unofficial dataloader for the [Sen2Venµs dataset](https://zenodo.org/record/6514159), baked at [CESBIO](https://www.cesbio.cnrs.fr/) by [Julien Michel, Juan VInasco-Salinas, Jordi Inglada and Olivier Hagolle](https://doi.org/10.3390/data7070096).

## Overview

This package provides a simple way to download and use the [Sen2Venµs dataset](https://zenodo.org/record/6514159) within the pytorch and Xarray ecosystems.

```python
from sen2venus import Sen2Venus
import matplotlib.pyplot as plt

Sen2Venus('./').download('SUDOUE-4')
dataset = Sen2Venus('./', load_geometry=True, subset='rededge')
input, target = dataset.getitem_xarray(idx)
input.plot.imshow(col='band')
target.plot.imshow(col='band')
plt.show()
```
![Matching Sentinel 2 and Venus samples](examples/samples_sentinel_venus.png)

## Features

- [x] **Automatic download from zenodo**: The Zenodo URLs and hashes are included. From a region name ([see the list](https://zenodo.org/record/6514159)), the corresponding subset is downloaded and decompressed. 

- [x] **x2 or x4 dataset loading**: you can pick the `rgbnir` or the `rededge` subset to load the x2 or the x4 low and high resolution patches.

- [x] **inspired from existing frameworks**: the Sen2Venus class is inspired from [the torchsr dataset definition style](https://github.com/Coloquinte/torchSR/tree/main/torchsr/datasets) and the torchvision download utility are used.

- [x] **automatically retrieve geospatial information**: includes method to convert the dataset samples to Xarray `DataArray`s

## TODO (WIP)

- [ ] (wip) better integration of download within class instantiation - currently needs to be reinstantiated
- [ ] (wip) parallel downloads / multiple regions download

## Installation

You can install the package using `pip`:

```bash
pip install sen2venus 
```

## Documentation (WIP)

For more detailed information on the available parameters, methods, and best practices, please refer to the documentation.

## Support and Issues (WIP)

If you encounter any issues, bugs, or have questions about the package, please feel free to open an issue on the GitHub repository. We appreciate your feedback!

## License

This package is released under the MIT License.