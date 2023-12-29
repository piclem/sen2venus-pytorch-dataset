# sen2venus
Unofficial dataloader for the [Sen2Venµs dataset](https://zenodo.org/record/6514159).

## Overview

(wip)

## Features

- **Automatic download from zenodo**: The Zenodo URLs and hashes are included. From a region name ([see the list](https://zenodo.org/record/6514159)), the corresponding subset is downloaded and decompressed. 

- **x2 or x4 dataset loading**: you can pick the `rgbnir` or the `rededge` subset to load the x2 or the x4 low and high resolution patches.

- **inspired from existing frameworks**: the Sen2Venus class is inspired from [the torchsr dataset definition style](https://github.com/Coloquinte/torchSR/tree/main/torchsr/datasets) and the torchvision download utility are used.

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