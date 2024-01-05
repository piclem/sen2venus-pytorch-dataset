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

class Sen2VenusURLs:
    """Helper class for Sen2Venus Dataset URLs"""
    urls = [('https://zenodo.org/record/6514159/files/ALSACE.7z?download=1', 'ecbf57fc83a8c8ca47ab421642bbef57'), ('https://zenodo.org/record/6514159/files/ANJI.7z?download=1', '2b6521e2fd43fc220557d1a171f94c06'), ('https://zenodo.org/record/6514159/files/ARM.7z?download=1', '9c264cd01640707f483f78a88c1a40c8'), ('https://zenodo.org/record/6514159/files/ATTO.7z?download=1', 'c6d7905816f8c807e5a87f4a2d09a4ae'), ('https://zenodo.org/record/6514159/files/BAMBENW2.7z?download=1', 'f804161f30c295dab1172e904ecb38be'), ('https://zenodo.org/record/6514159/files/BENGA.7z?download=1', 'a3bdc8fd5ac049b2d07b308fc1f0706a'), ('https://zenodo.org/record/6514159/files/ES-IC3XG.7z?download=1', 'e7a19cd51f048a006688f6b2ea795d55'), ('https://zenodo.org/record/6514159/files/ES-LTERA.7z?download=1', '226cd7c10689f9aad92c760d9c1899fe'), ('https://zenodo.org/record/6514159/files/ESGISB-1.7z?download=1', 'ab1c0e9a70c566d6fe8b94ba421a15d6'), ('https://zenodo.org/record/6514159/files/ESGISB-2.7z?download=1', '20196e6e963170e641fc805330077434'), ('https://zenodo.org/record/6514159/files/ESGISB-3.7z?download=1', 'ac42ab2ddb89975b55395ace90ecc0a6'), ('https://zenodo.org/record/6514159/files/ESTUAMAR.7z?download=1', '2b540369499c7b9882f7e195699e9438'), ('https://zenodo.org/record/6514159/files/FGMANAUS.7z?download=1', '06d422d9f4ba0c2ed1087c2a7f0339c5'), ('https://zenodo.org/record/6514159/files/FR-BIL.7z?download=1', 'c4305e091b61de5583842f71b4122ed3'), ('https://zenodo.org/record/6514159/files/FR-LAM.7z?download=1', '1bceb23259d7f101ee0e1df141b5e550'), ('https://zenodo.org/record/6514159/files/FR-LQ1.7z?download=1', '535489d0d3bc23e8e7646a20b99575e6'), ('https://zenodo.org/record/6514159/files/JAM2018.7z?download=1', '2e2a6de2b5842ce86d074ebd8c68354b'), ('https://zenodo.org/record/6514159/files/K34-AMAZ.7z?download=1', '7abf9ef3f89bd30b905c0029169b88d1'), ('https://zenodo.org/record/6514159/files/KUDALIAR.7z?download=1', '1427c8a4bc1e238c5c63e434fd6d31c6'), ('https://zenodo.org/record/6514159/files/LERIDA-1.7z?download=1', 'd507dcbc1b92676410df9e4f650ea23b'), ('https://zenodo.org/record/6514159/files/LICENCE?download=1', '373f2ea88a57d51c5f54778c36503027'), ('https://zenodo.org/record/6514159/files/MAD-AMBO.7z?download=1', '49e43cd47ecdc5360c83e448eaf73fbb'), ('https://zenodo.org/record/6514159/files/MD5SUMS?download=1', 'a21a655812d6cfd309d1e76c95463916'), ('https://zenodo.org/record/6514159/files/NARYN.7z?download=1', '56474220d0014e53aa0c96ea93c03bc9'), ('https://zenodo.org/record/6514159/files/SO1.7z?download=1', '62b5ce44dc641639079c15227cdbd794'), ('https://zenodo.org/record/6514159/files/SO2.7z?download=1', '59afd969b950f90df0f8ce8b1dbccd62'), ('https://zenodo.org/record/6514159/files/SUDOUE-2.7z?download=1', '5aed36a3d5e9746e5f5c438d10fae413'), ('https://zenodo.org/record/6514159/files/SUDOUE-3.7z?download=1', '0eeb556caaae171b8fbd0696f4757308'), ('https://zenodo.org/record/6514159/files/SUDOUE-4.7z?download=1', 'aac762b62ac240720d34d5bb3fc4a906'), ('https://zenodo.org/record/6514159/files/SUDOUE-5.7z?download=1', '69042546af7bd25a0398b04c2ce60057'), ('https://zenodo.org/record/6514159/files/SUDOUE-6.7z?download=1', 'ca143d2a2a56db30ab82c33420433e01')]
    
    def get_url(self, site):
        for url, md5 in self.urls:
            if site in url:
                return url
        raise ValueError(f'Site {site} not found in urls')
    
    def get_sites_list(self):
        return [self.url.split('/')[-1].split('.7z')[0] for url, md5 in self.urls]

class Sen2VenusSubsetSuffixes():
    """data class for suffixes"""
    sen_rgbnir = '_10m_b2b3b4b8.pt'
    sen_rededge = '_20m_b4b5b6b8a.pt'
    venus_rgbnir ='_05m_b2b3b4b8.pt'
    venus_rededge = '_05m_b4b5b6b8a.pt'

class Sen2VenusSite(Dataset):
    def __init__(self, root, site_name, load_geometry=False, subset='all', force_download=False):
        """Class for single Site Dataset

        Args:
            root (_type_): local download root path
            site_name (_type_): Venus site name. Use Sen2VenusURLs().get_sites_list() for a complete list of available sites.
            load_geometry (bool, optional): return the geospatial geometry for each batch. Defaults to False.
            subset (str, optional): bands subset, one of ('rgbnir', 'rededge', 'all'). Defaults to 'all'.
            force_download (bool, optional): Force download and extraction even if files already exist locally in the root folder. Defaults to False.

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
        
        self.download(force_download)
        self.parse()

    def parse(self):
        """Parse the downloaded files and create a list of samples"""
        self.total_samples = 0
        self.samples = []

        pt_files = sorted(self.site_root.glob('*.pt'))
        unique_prefixes = set(['_'.join(f.stem.split('_')[:3]) for f in pt_files])

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
        tensor = torch.load(file_path)
        return tensor.size(0)
    
    def is_already_downloaded(self):
        return self.site_root.with_suffix('.7z').exists()
    
    def is_already_extracted(self):
        return self.site_root.is_dir() and self.is_folder_valid()

    def is_folder_valid(self):
        """check wether the folder contains index.csv, LICENCE, 4*N .pt files and N .gpkg files"""
        if not (self.site_root / 'index.csv').exists():
            return False
        if not (self.site_root / 'LICENCE').exists():
            return False
        num_pt = len(list(self.site_root.glob(f'*.pt')))
        num_gpkg = len(list(self.site_root.glob(f'*.gpkg')))
        return num_pt > 0 and num_gpkg > 0 and num_pt == 4*num_gpkg

    def download(self, force_download=False):
        import py7zr
        if (not self.is_already_downloaded()) or force_download:
            filename = None
            md5sum = None
            if isinstance(self.url, str):
                url = self.url
                if not filename:
                    filename = os.path.basename(url)
            else:
                url = self.url[0]
                if len(self.url) > 1:
                    md5sum = self.url[1]
                if len(self.url) > 2:
                    filename = self.url[2]
                else:
                    filename = os.path.basename(url)[:-11]
            download_url(self.url, self.root, filename=filename, md5=md5sum)
        if (not self.is_already_extracted()) or force_download: 
            with py7zr.SevenZipFile(Path(self.root) / filename, mode='r') as z:
                if not (Path(self.root) / filename).with_suffix('').exists():
                    print('Extracting 7zip archive')
                    z.extractall()
        # self.already_downloaded_urls.append((self.root, url))
        # self.update()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        input_files, target_files, batch_pos = self.samples[idx]
        if self.subset == 'rgbnir':
            input_tensor = torch.load(input_files[0])[batch_pos]/self.SCALE
            target_tensor = torch.load(target_files[0])[batch_pos]/self.SCALE
        if self.subset == 'rededge':
            input_tensor = torch.load(input_files[1])[batch_pos]/self.SCALE
            target_tensor = torch.load(target_files[1])[batch_pos]/self.SCALE
        if self.subset == 'all':
            # load 20m sen rededge and upscale using bicubic interpolation
            input_tensor = torch.concat(
                (
                    torch.load(input_files[0])[batch_pos]/self.SCALE,
                    torch.nn.functional.interpolate(torch.load(input_files[1])[batch_pos].unsqueeze(0)/self.SCALE, scale_factor=(2,2), mode='bicubic').squeeze(0),
                ), dim=0)
            target_tensor = torch.concat(
                (
                    torch.load(target_files[0])[batch_pos]/self.SCALE, 
                    torch.load(target_files[1])[batch_pos]/self.SCALE
                ), dim=0)
            
        
        if self.load_geometry:
            geometry = gpd.read_file(self.find_matching_gpkg(input_files[0]), rows=slice(batch_pos, batch_pos+1))
            return input_tensor, target_tensor, (geometry.to_json(), geometry.crs)
        return input_tensor, target_tensor
    
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
    def __init__(self, root, site_names=[], load_geometry=False, subset='rgbnir', ):
        # create Sen2VenusSite list
        self.subset = subset
        self.datasets = []
        for site_name in site_names:
            self.datasets.append(Sen2VenusSite(root, site_name, load_geometry=load_geometry, subset=subset))
        super().__init__(self.datasets)


