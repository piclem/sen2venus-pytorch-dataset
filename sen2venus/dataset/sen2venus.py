import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.utils import download_url
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path
import json
import shutil
from tqdm.contrib.concurrent import thread_map


def _build_augmentations():
    """Build the Kornia augmentation pipeline (random 90° rotation + flips).

    Shared by both the v1 and v2 datasets. Kept lazy so that ``kornia`` is only
    imported when ``augment=True``.
    """
    import kornia as K
    from kornia.augmentation import AugmentationSequential

    # from https://github.com/kornia/kornia/issues/1531
    class RandomRot90(K.augmentation.AugmentationBase2D):
        def generate_parameters(self, input_shape):
            assert len(input_shape) == 4
            assert input_shape[-1] == input_shape[-2], input_shape  # has to be square
            k = torch.randint(0, 3, (input_shape[0],))
            return dict(k=k)

        def apply_transform(self, input, params, *args, **kwargs):
            outputs = [torch.rot90(item, k, dims=(-1, -2)) for item, k in zip(input, params['k'])]
            return torch.stack(outputs, dim=0)

    return AugmentationSequential(
        RandomRot90(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),
        K.augmentation.RandomHorizontalFlip(p=0.5),
        same_on_batch=False,
    )


class Sen2VenusURLs:
    """Helper class for Sen2Venus Dataset URLs"""
    urls = [('https://zenodo.org/record/6514159/files/ALSACE.7z?download=1', 'ecbf57fc83a8c8ca47ab421642bbef57'), ('https://zenodo.org/record/6514159/files/ANJI.7z?download=1', '2b6521e2fd43fc220557d1a171f94c06'), ('https://zenodo.org/record/6514159/files/ARM.7z?download=1', '9c264cd01640707f483f78a88c1a40c8'), ('https://zenodo.org/record/6514159/files/ATTO.7z?download=1', 'c6d7905816f8c807e5a87f4a2d09a4ae'), ('https://zenodo.org/record/6514159/files/BAMBENW2.7z?download=1', 'f804161f30c295dab1172e904ecb38be'), ('https://zenodo.org/record/6514159/files/BENGA.7z?download=1', 'a3bdc8fd5ac049b2d07b308fc1f0706a'), ('https://zenodo.org/record/6514159/files/ES-IC3XG.7z?download=1', 'e7a19cd51f048a006688f6b2ea795d55'), ('https://zenodo.org/record/6514159/files/ES-LTERA.7z?download=1', '226cd7c10689f9aad92c760d9c1899fe'), ('https://zenodo.org/record/6514159/files/ESGISB-1.7z?download=1', 'ab1c0e9a70c566d6fe8b94ba421a15d6'), ('https://zenodo.org/record/6514159/files/ESGISB-2.7z?download=1', '20196e6e963170e641fc805330077434'), ('https://zenodo.org/record/6514159/files/ESGISB-3.7z?download=1', 'ac42ab2ddb89975b55395ace90ecc0a6'), ('https://zenodo.org/record/6514159/files/ESTUAMAR.7z?download=1', '2b540369499c7b9882f7e195699e9438'), ('https://zenodo.org/record/6514159/files/FGMANAUS.7z?download=1', '06d422d9f4ba0c2ed1087c2a7f0339c5'), ('https://zenodo.org/record/6514159/files/FR-BIL.7z?download=1', 'c4305e091b61de5583842f71b4122ed3'), ('https://zenodo.org/record/6514159/files/FR-LAM.7z?download=1', '1bceb23259d7f101ee0e1df141b5e550'), ('https://zenodo.org/record/6514159/files/FR-LQ1.7z?download=1', '535489d0d3bc23e8e7646a20b99575e6'), ('https://zenodo.org/record/6514159/files/JAM2018.7z?download=1', '2e2a6de2b5842ce86d074ebd8c68354b'), ('https://zenodo.org/record/6514159/files/K34-AMAZ.7z?download=1', '7abf9ef3f89bd30b905c0029169b88d1'), ('https://zenodo.org/record/6514159/files/KUDALIAR.7z?download=1', '1427c8a4bc1e238c5c63e434fd6d31c6'), ('https://zenodo.org/record/6514159/files/LERIDA-1.7z?download=1', 'd507dcbc1b92676410df9e4f650ea23b'), ('https://zenodo.org/record/6514159/files/MAD-AMBO.7z?download=1', '49e43cd47ecdc5360c83e448eaf73fbb'), ('https://zenodo.org/record/6514159/files/NARYN.7z?download=1', '56474220d0014e53aa0c96ea93c03bc9'), ('https://zenodo.org/record/6514159/files/SO1.7z?download=1', '62b5ce44dc641639079c15227cdbd794'), ('https://zenodo.org/record/6514159/files/SO2.7z?download=1', '59afd969b950f90df0f8ce8b1dbccd62'), ('https://zenodo.org/record/6514159/files/SUDOUE-2.7z?download=1', '5aed36a3d5e9746e5f5c438d10fae413'), ('https://zenodo.org/record/6514159/files/SUDOUE-3.7z?download=1', '0eeb556caaae171b8fbd0696f4757308'), ('https://zenodo.org/record/6514159/files/SUDOUE-4.7z?download=1', 'aac762b62ac240720d34d5bb3fc4a906'), ('https://zenodo.org/record/6514159/files/SUDOUE-5.7z?download=1', '69042546af7bd25a0398b04c2ce60057'), ('https://zenodo.org/record/6514159/files/SUDOUE-6.7z?download=1', 'ca143d2a2a56db30ab82c33420433e01')]
    
    def get_url(self, site):
        for url, md5 in self.urls:
            if site in url:
                return url, md5
        raise ValueError(f'Site {site} not found in urls')
    
    def get_sites_list(self):
        return [url.split('/')[-1].split('.7z')[0] for url, md5 in self.urls]

class Sen2VenusSubsetSuffixes():
    """data class for suffixes"""
    sen_rgbnir = '_10m_b2b3b4b8.pt'
    sen_rededge = '_20m_b4b5b6b8a.pt'
    venus_rgbnir ='_05m_b2b3b4b8.pt'
    venus_rededge = '_05m_b4b5b6b8a.pt'

class Sen2VenusSite(Dataset):
    def __init__(self, root, site_name, load_geometry=False, subset='all', force_download=False, re_upsampling_mode='bicubic', return_type='tuple', device='cpu', augment=False):
        """Class for single Site Dataset

        Args:
            root (_type_): local download root path
            site_name (_type_): Venus site name. Use Sen2VenusURLs().get_sites_list() for a complete list of available sites.
            load_geometry (bool, optional): return the geospatial geometry for each batch. Defaults to False.
            subset (str, optional): bands subset, one of ('rgbnir', 'rededge', 'all'). Defaults to 'all'.
            force_download (bool, optional): Force download and extraction even if files already exist locally in the root folder. Defaults to False.
            re_upsampling_mode (str, optional): upsampling mode for the red edge bands (20m > 10m). Defaults to 'bicubic'.
            return_type (str, optional): return type of the __getitem__ method. Defaults to 'tuple'. use 'dict' to return a dictionary with the keys 'lr' and 'hr'
            device (str, optional): device to load the data to. Defaults to 'cpu'.
            augment (bool, optional): apply Kornia augmentations to the data (flip and 90 rotation). Defaults to False.

        Raises:
            NotImplementedError: if subset is not in ('rgbnir','rededge', 'all')
        """
        self.url = Sen2VenusURLs().get_url(site_name)
        self.site_name = site_name
        self.root = root
        self.site_root = Path(self.root) / self.site_name
        if subset not in ('rgbnir', 'rededge', 'all'): 
            raise NotImplementedError(f'Subset "{subset}" not implemented')
        self.subset = subset
        self.SCALE = 10000.0
        self.samples = []
        self.total_samples = 0
        self.load_geometry = load_geometry
        self.return_type = return_type
        self.device = device
        self.augment = augment
        if augment:
            self.augmentations = _build_augmentations()
        self.re_upsampling_mode = re_upsampling_mode
        
        self.download(force_download)
        for attempt in range(2):
            try:
                self.parse()
            except Exception as e:
                if attempt > 0:
                    print(f"Download failed for {self.site_name} - aborting.\nException: {e}")
                    raise
                print(f'attempt {attempt+1} failed, trying re-download.')
                self.download(force_download=True)
                continue
            break


    def parse(self):
        """Parse the downloaded files and create a list of samples"""
        self.total_samples = 0
        self.samples = []

        pt_files = sorted(self.site_root.glob('*.pt'))
        unique_prefixes = set(['_'.join(f.stem.split('_')[:-2]) for f in pt_files])

        for p in unique_prefixes:
            input_files = [
                self.site_root / (p + Sen2VenusSubsetSuffixes.sen_rgbnir),
                self.site_root / (p + Sen2VenusSubsetSuffixes.sen_rededge)
            ]
            target_files = [
                self.site_root / (p + Sen2VenusSubsetSuffixes.venus_rgbnir),
                self.site_root / (p + Sen2VenusSubsetSuffixes.venus_rededge)
            ]

            
            b = self.get_num_samples(input_files[0])
            for batch_pos in range(b):
                self.samples.append((input_files, target_files, batch_pos))
            self.total_samples += b
    
    def find_matching_gpkg(self, input_file):
        """Find the matching gpkg file for a given input file

        Args:
            input_file (str): path to a .pt file

        Returns:
            str: path to corresponding .gpkg file
        """
        matching_gpkg = '_'.join(str(input_file).split('_')[:-2])+"_patches.gpkg"
        return matching_gpkg

    def get_num_samples(self, file_path):
        """ Get the number of samples in batch dimension in a given .pt file

        Args:
            file_path (str): path to .pt file

        Returns:
            int: number of batches or dimension 0 size
        """
        tensor = torch.load(file_path, map_location='cpu', mmap=True, weights_only=True)
        return tensor.size(0)
    
    def is_already_downloaded(self):
        return self.site_root.with_suffix('.7z').exists()
    
    def is_already_extracted(self):
        return self.site_root.is_dir() and self.is_folder_valid()

    def is_folder_valid(self, strict=False):
        """Check wether the folder contains index.csv, LICENCE, 4*N .pt files and N .gpkg files, unless strict=False for the latest"""
        if not (self.site_root / 'index.csv').exists():
            return False
        if not (self.site_root / 'LICENCE').exists():
            return False
        num_pt = len(list(self.site_root.glob(f'*.pt')))
        num_gpkg = len(list(self.site_root.glob(f'*.gpkg')))
        return num_pt > 0 and num_gpkg > 0 and (num_pt == 4*num_gpkg or not strict)

    def download(self, force_download=False):
        import py7zr
        # self.url may be a bare URL string or a (url, md5[, filename]) tuple.
        # Resolve url/md5sum/filename unconditionally so the extract-only path
        # (archive already downloaded but not yet extracted) has a valid filename.
        if isinstance(self.url, str):
            url = self.url
            md5sum = None
            filename = os.path.basename(url)
        else:
            url = self.url[0]
            md5sum = self.url[1] if len(self.url) > 1 else None
            filename = self.url[2] if len(self.url) > 2 else os.path.basename(url)[:-11]

        if not (self.is_already_extracted() or self.is_already_downloaded()) or force_download:
            download_url(url, self.root, filename=filename, md5=md5sum)
        if (not self.is_already_extracted()) or force_download:
            f = Path(self.root) / filename
            with py7zr.SevenZipFile(f, mode='r') as z:
                target = (Path(self.root) / filename).with_suffix('')
                if target.exists():
                    print(f"Removing existing folder: {f}")
                    shutil.rmtree(target)
                print('Extracting 7zip archive')
                z.extractall(self.root)
                    
            f.unlink()
                    
        # self.already_downloaded_urls.append((self.root, url))
        # self.update()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        input_files, target_files, batch_pos = self.samples[idx]
        if self.subset == 'rgbnir':
            input_tensor = torch.load(input_files[0], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE
            target_tensor = torch.load(target_files[0], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE
        if self.subset == 'rededge':
            input_tensor = torch.load(input_files[1], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE
            target_tensor = torch.load(target_files[1], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE
        if self.subset == 'all':
            # load 20m sen rededge and upscale using bicubic interpolation
            input_tensor = torch.concat(
                (
                    torch.load(input_files[0], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE,
                    torch.nn.functional.interpolate(torch.load(input_files[1], map_location=self.device, mmap=True, weights_only=True)[batch_pos].unsqueeze(0)/self.SCALE, scale_factor=(2,2), mode=self.re_upsampling_mode).squeeze(0),
                ), dim=0)
            target_tensor = torch.concat(
                (
                    torch.load(target_files[0], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE, 
                    torch.load(target_files[1], map_location=self.device, mmap=True, weights_only=True)[batch_pos]/self.SCALE
                ), dim=0)
        # augmentations
        if self.augment:
            with torch.no_grad():
                input_tensor = self.augmentations(input_tensor)[0]
                target_tensor = self.augmentations(target_tensor, params=self.augmentations._params)[0]
        
        if self.load_geometry:
            geometry = gpd.read_file(self.find_matching_gpkg(input_files[0]), rows=slice(batch_pos, batch_pos+1))
            return input_tensor, target_tensor, (geometry.to_json(), geometry.crs)
        if self.return_type == 'tuple':
            return input_tensor, target_tensor
        elif self.return_type == 'dict':
            return {'lr': input_tensor, 'hr': target_tensor}
        else:
            raise NotImplementedError(f'Return type "{self.return_type}" not implemented')
    
    def getitem_xarray(self, idx):
        assert self.load_geometry, "Cannot use `getitem_xarray()` if `load_geometry` is False, use `load_geometry = True` when instantiating the dataset."
        inputs, targets, geometry = self.__getitem__(idx)
        gdf = gpd.GeoDataFrame.from_features(json.loads(geometry[0]), crs=geometry[1])
        
        minx,miny,maxx,maxy = gdf.total_bounds
        # inputs
        gsd = (maxx-minx) / (inputs.shape[-1])
        xs = np.arange(minx, maxx, gsd)+gsd/2
        ys = np.arange(maxy, miny, -gsd)-gsd/2
        if self.subset == 'rgbnir':
            band_names = ['b2','b3','b4','b8']
        elif self.subset == 'rededge':
            band_names = ['re1','re2','re3','re4 (b8a)']
        elif self.subset == 'all':
            band_names = ['b2','b3','b4','b8','re1','re2','re3','re4 (b8a)']
        else:
            raise NotImplementedError(f'Subset "{self.subset}" not implemented')
        da_input = xr.DataArray(inputs, dims=['band', 'y', 'x'], coords={'band': band_names, 'y':ys, 'x':xs })
        da_input = da_input.rio.write_crs(gdf.crs)

        # targets
        gsd = (maxx-minx) / (targets.shape[-1])
        xs = np.arange(minx, maxx, gsd)+gsd/2
        ys = np.arange(maxy, miny, -gsd)-gsd/2
        da_target = xr.DataArray(targets, dims=['band', 'y', 'x'], coords={'band': band_names, 'y':ys, 'x':xs })
        da_target = da_target.rio.write_crs(gdf.crs)

        return da_input, da_target

class Sen2Venus(ConcatDataset):
    def __init__(self, root, site_names=[], subset='rgbnir', **kwargs):
        # create Sen2VenusSite list
        self.subset = subset
        self.datasets = []
        
        def _get_sen2venus_site_instance(site_name):
            return Sen2VenusSite(root, site_name, subset=subset, **kwargs)

        self.datasets = thread_map(_get_sen2venus_site_instance, site_names)
        super().__init__(self.datasets)


class Sen2VenusV2(Dataset):
    """PyTorch dataset for the Sen2Venµs **v2.0** release.

    Unlike the v1 dataset (stacked ``.pt`` tensors downloaded from Zenodo), v2 is
    distributed as per-patch GeoTIFFs and accessed through ``tacoreader`` from the
    `tacofoundation <https://huggingface.co/datasets/tacofoundation/sen2venus>`_
    mirror. ``tacoreader`` transparently handles streaming/caching of the remote
    GeoTIFFs, so this class is a thin adapter that turns each sample into
    ``(lr, hr)`` torch tensors with the same conventions as the v1
    :class:`Sen2VenusSite` (reflectance scaled to ``[0, 1]``, ``subset`` band
    selection, optional Kornia augmentation, xarray export).

    .. important::
        The ``tacofoundation:sen2venus`` dataset uses the legacy "tortilla"
        format, which is **only readable by** ``tacoreader < 1.0``. tacoreader
        2.0+ dropped support for it. Install the right version with
        ``pip install 'sen2venus-pytorch-dataset[v2]'``.

    Sample geometry (verified live): LR (Sentinel-2) is 128×128 @ 10 m with 10
    bands; HR (Venµs) is 256×256 @ 5 m with 8 bands; ~124k patches over 28 sites,
    all in the ``train`` split.

    Key differences vs v1 to be aware of:

    - **SWIR bands** (B11, B12) are available in v2, but **only on the Sentinel-2
      (LR) side** — Venµs has no SWIR. They are *off by default* to match v1; set
      ``include_swir=True`` to append them to the LR tensor. The HR tensor is
      unaffected, so when SWIR is enabled the LR and HR tensors have a different
      number of bands (LR = subset + 2, HR = subset).
    - **Band order** follows the native v2 GeoTIFF order
      ``[B2, B3, B4, B5, B6, B7, B8, B8A, (B11, B12)]``, which differs from v1's
      ``all`` ordering. The selected band names are exposed on ``self.lr_bands`` /
      ``self.hr_bands``.
    - No per-band upsampling is needed: each LR GeoTIFF is already a single
      harmonized grid, so the v1 ``re_upsampling_mode`` argument does not apply.

    Args:
        source (str, optional): tacoreader dataset identifier. Defaults to
            ``"tacofoundation:sen2venus"``.
        site_names (str | list[str] | None, optional): restrict the dataset to one
            or more sites (e.g. ``"SUDOUE-4"``). Matched against the site prefix of
            each sample id. See :meth:`available_sites`. Defaults to ``None`` (all
            28 sites).
        subset (str, optional): band subset, one of ``('rgbnir', 'rededge',
            'all')``. Defaults to ``'all'`` (same as v1).
        include_swir (bool, optional): append the SWIR bands B11 & B12 to the LR
            tensor. Only meaningful for the Sentinel-2 side. Defaults to ``False``.
        return_type (str, optional): ``'tuple'`` -> ``(lr, hr)`` or ``'dict'`` ->
            ``{'lr': ..., 'hr': ...}``. Defaults to ``'tuple'``.
        device (str, optional): device to load tensors onto. Defaults to ``'cpu'``.
        augment (bool, optional): apply Kornia augmentations (flips + 90° rotation).
            Defaults to ``False``.
        scale (float, optional): reflectance scaling divisor. Defaults to
            ``10000.0``.
        nodata (int, optional): no-data value to mask to 0 before scaling. Defaults
            to ``65535`` (the v2 GeoTIFF no-data value).
    """

    # Native band order of the v2 GeoTIFFs (1-based indices used by rasterio).
    LR_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    HR_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A']
    SWIR_BANDS = ['B11', 'B12']
    SUBSETS = {
        'rgbnir': ['B2', 'B3', 'B4', 'B8'],
        'rededge': ['B5', 'B6', 'B7', 'B8A'],
        'all': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A'],
    }

    def __init__(self, source="tacofoundation:sen2venus", site_names=None, subset='all',
                 include_swir=False, return_type='tuple', device='cpu', augment=False,
                 scale=10000.0, nodata=65535):
        if subset not in self.SUBSETS:
            raise NotImplementedError(
                f'Subset "{subset}" not implemented, use one of {tuple(self.SUBSETS)}'
            )
        if return_type not in ('tuple', 'dict'):
            raise NotImplementedError(f'Return type "{return_type}" not implemented')

        self.source = source
        self.subset = subset
        self.include_swir = include_swir
        self.return_type = return_type
        self.device = device
        self.augment = augment
        self.scale = scale
        self.nodata = nodata

        # Resolve the band names / 1-based rasterio indices for each side.
        self.lr_bands = list(self.SUBSETS[subset]) + (self.SWIR_BANDS if include_swir else [])
        self.hr_bands = list(self.SUBSETS[subset])
        self._lr_idx = [self.LR_BANDS.index(b) + 1 for b in self.lr_bands]
        self._hr_idx = [self.HR_BANDS.index(b) + 1 for b in self.hr_bands]

        self.taco = self._load(source)
        self.indices = self._filter_sites(site_names)

        if augment:
            self.augmentations = _build_augmentations()

    @staticmethod
    def _load(source):
        """Load the tortilla dataframe, with friendly errors for the common pitfalls."""
        try:
            import tacoreader
        except ImportError as e:
            raise ImportError(
                "Sen2VenusV2 requires the 'tacoreader' package (and fsspec/requests/"
                "aiohttp for remote reads). Install it with: "
                "pip install 'sen2venus-pytorch-dataset[v2]'"
            ) from e
        try:
            return tacoreader.load(source)
        except Exception as e:
            # tacoreader >= 1.0 dropped support for the legacy 'tortilla' format
            # that tacofoundation:sen2venus uses, raising a format error here.
            version = getattr(tacoreader, "__version__", "?")
            raise RuntimeError(
                f"Failed to load '{source}' with tacoreader {version}. "
                "The tacofoundation:sen2venus dataset uses the legacy tortilla "
                "format, which is only readable by tacoreader < 1.0. "
                "Install a compatible version with: "
                "pip install 'sen2venus-pytorch-dataset[v2]' "
                "(or: pip install 'tacoreader<1.0' fsspec requests aiohttp)."
            ) from e

    @staticmethod
    def _site_of(sample_id):
        """Site name is the prefix of the sample id, e.g. 'SUDOUE-4_225_...' -> 'SUDOUE-4'."""
        return str(sample_id).split('_')[0]

    def available_sites(self):
        """Return the sorted list of site names present in the loaded dataset."""
        return sorted({self._site_of(i) for i in self.taco["tortilla:id"]})

    def _filter_sites(self, site_names):
        """Return the list of underlying row indices, optionally filtered by site."""
        n = len(self.taco)
        if not site_names:
            return list(range(n))
        if isinstance(site_names, str):
            site_names = [site_names]
        site_names = set(site_names)
        ids = [str(v) for v in self.taco["tortilla:id"]]
        indices = [i for i, v in enumerate(ids) if self._site_of(v) in site_names]
        if not indices:
            available = sorted({self._site_of(v) for v in ids})
            raise ValueError(
                f"No samples found for sites {sorted(site_names)}. "
                f"Available sites: {available}"
            )
        return indices

    def _read_src(self, src, band_indices):
        """Read selected bands from an open rasterio dataset as scaled float32."""
        arr = src.read(band_indices).astype(np.float32)
        if self.nodata is not None:
            arr[arr == self.nodata] = 0.0
        arr /= self.scale
        return arr

    def _load_arrays(self, idx):
        """Load the (lr, hr) arrays and their geotransform/crs for a sample."""
        import rasterio as rio
        row = self.taco.read(self.indices[idx])
        lr_path, hr_path = row.read(0), row.read(1)
        with rio.open(lr_path) as src_lr, rio.open(hr_path) as src_hr:
            lr = self._read_src(src_lr, self._lr_idx)
            hr = self._read_src(src_hr, self._hr_idx)
            lr_meta = (src_lr.transform, src_lr.crs)
            hr_meta = (src_hr.transform, src_hr.crs)
        return lr, hr, lr_meta, hr_meta

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        lr, hr, _, _ = self._load_arrays(idx)
        input_tensor = torch.from_numpy(lr).to(self.device)
        target_tensor = torch.from_numpy(hr).to(self.device)

        if self.augment:
            with torch.no_grad():
                input_tensor = self.augmentations(input_tensor)[0]
                target_tensor = self.augmentations(target_tensor, params=self.augmentations._params)[0]

        if self.return_type == 'tuple':
            return input_tensor, target_tensor
        return {'lr': input_tensor, 'hr': target_tensor}

    def _to_dataarray(self, arr, meta, band_names):
        transform, crs = meta
        _, height, width = arr.shape
        # pixel-centre coordinates from the affine geotransform
        xs = transform.c + transform.a * (np.arange(width) + 0.5)
        ys = transform.f + transform.e * (np.arange(height) + 0.5)
        da = xr.DataArray(arr, dims=['band', 'y', 'x'],
                          coords={'band': band_names, 'y': ys, 'x': xs})
        return da.rio.write_crs(crs)

    def getitem_xarray(self, idx):
        """Return the (lr, hr) sample as georeferenced xarray ``DataArray``s."""
        lr, hr, lr_meta, hr_meta = self._load_arrays(idx)
        da_input = self._to_dataarray(lr, lr_meta, self.lr_bands)
        da_target = self._to_dataarray(hr, hr_meta, self.hr_bands)
        return da_input, da_target


