import os
import torch
from torch.utils.data import Dataset
import torchvision
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path
import logging
import json

class Sen2Venus(Dataset):
    def __init__(self, root, load_geometry=False, subset='rgbnir', **kwargs):
        
        if subset=='rgbnir':
            self.bands_pattern = 'b2b3b4b8'
            self.input_gsd_pattern = '10m'
        elif subset=='rededge':
            self.bands_pattern = 'b4b5b6b8a'
            self.input_gsd_pattern = '20m'
        else:
            raise NotImplementedError(f'Subset "{subset}" not implemented')
        self.SCALE = 10000.0
        self.samples = []
        self.root = root
        self.total_samples = 0
        self.load_geometry = load_geometry
        self.already_downloaded_urls = []
        self.urls = [('https://zenodo.org/record/6514159/files/ALSACE.7z?download=1', 'ecbf57fc83a8c8ca47ab421642bbef57'), ('https://zenodo.org/record/6514159/files/ANJI.7z?download=1', '2b6521e2fd43fc220557d1a171f94c06'), ('https://zenodo.org/record/6514159/files/ARM.7z?download=1', '9c264cd01640707f483f78a88c1a40c8'), ('https://zenodo.org/record/6514159/files/ATTO.7z?download=1', 'c6d7905816f8c807e5a87f4a2d09a4ae'), ('https://zenodo.org/record/6514159/files/BAMBENW2.7z?download=1', 'f804161f30c295dab1172e904ecb38be'), ('https://zenodo.org/record/6514159/files/BENGA.7z?download=1', 'a3bdc8fd5ac049b2d07b308fc1f0706a'), ('https://zenodo.org/record/6514159/files/ES-IC3XG.7z?download=1', 'e7a19cd51f048a006688f6b2ea795d55'), ('https://zenodo.org/record/6514159/files/ES-LTERA.7z?download=1', '226cd7c10689f9aad92c760d9c1899fe'), ('https://zenodo.org/record/6514159/files/ESGISB-1.7z?download=1', 'ab1c0e9a70c566d6fe8b94ba421a15d6'), ('https://zenodo.org/record/6514159/files/ESGISB-2.7z?download=1', '20196e6e963170e641fc805330077434'), ('https://zenodo.org/record/6514159/files/ESGISB-3.7z?download=1', 'ac42ab2ddb89975b55395ace90ecc0a6'), ('https://zenodo.org/record/6514159/files/ESTUAMAR.7z?download=1', '2b540369499c7b9882f7e195699e9438'), ('https://zenodo.org/record/6514159/files/FGMANAUS.7z?download=1', '06d422d9f4ba0c2ed1087c2a7f0339c5'), ('https://zenodo.org/record/6514159/files/FR-BIL.7z?download=1', 'c4305e091b61de5583842f71b4122ed3'), ('https://zenodo.org/record/6514159/files/FR-LAM.7z?download=1', '1bceb23259d7f101ee0e1df141b5e550'), ('https://zenodo.org/record/6514159/files/FR-LQ1.7z?download=1', '535489d0d3bc23e8e7646a20b99575e6'), ('https://zenodo.org/record/6514159/files/JAM2018.7z?download=1', '2e2a6de2b5842ce86d074ebd8c68354b'), ('https://zenodo.org/record/6514159/files/K34-AMAZ.7z?download=1', '7abf9ef3f89bd30b905c0029169b88d1'), ('https://zenodo.org/record/6514159/files/KUDALIAR.7z?download=1', '1427c8a4bc1e238c5c63e434fd6d31c6'), ('https://zenodo.org/record/6514159/files/LERIDA-1.7z?download=1', 'd507dcbc1b92676410df9e4f650ea23b'), ('https://zenodo.org/record/6514159/files/LICENCE?download=1', '373f2ea88a57d51c5f54778c36503027'), ('https://zenodo.org/record/6514159/files/MAD-AMBO.7z?download=1', '49e43cd47ecdc5360c83e448eaf73fbb'), ('https://zenodo.org/record/6514159/files/MD5SUMS?download=1', 'a21a655812d6cfd309d1e76c95463916'), ('https://zenodo.org/record/6514159/files/NARYN.7z?download=1', '56474220d0014e53aa0c96ea93c03bc9'), ('https://zenodo.org/record/6514159/files/SO1.7z?download=1', '62b5ce44dc641639079c15227cdbd794'), ('https://zenodo.org/record/6514159/files/SO2.7z?download=1', '59afd969b950f90df0f8ce8b1dbccd62'), ('https://zenodo.org/record/6514159/files/SUDOUE-2.7z?download=1', '5aed36a3d5e9746e5f5c438d10fae413'), ('https://zenodo.org/record/6514159/files/SUDOUE-3.7z?download=1', '0eeb556caaae171b8fbd0696f4757308'), ('https://zenodo.org/record/6514159/files/SUDOUE-4.7z?download=1', 'aac762b62ac240720d34d5bb3fc4a906'), ('https://zenodo.org/record/6514159/files/SUDOUE-5.7z?download=1', '69042546af7bd25a0398b04c2ce60057'), ('https://zenodo.org/record/6514159/files/SUDOUE-6.7z?download=1', 'ca143d2a2a56db30ab82c33420433e01')]
    

        for folder_name in os.listdir(self.root):
            folder_path = os.path.join(self.root, folder_name)
            if os.path.isdir(folder_path):
                input_files = []
                target_files = []

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith(f"_{self.input_gsd_pattern}_{self.bands_pattern}.pt"):
                        input_files.append(file_path)
                    elif file_name.endswith(f"_05m_{self.bands_pattern}.pt"):
                        target_files.append(file_path)

                for input_file in input_files:
                    target_file = self.find_matching_target_file(input_file, target_files)
                    if target_file:
                        b = self.get_num_samples(input_file)
                        for batch_pos in range(b):
                            self.samples.append((input_file, target_file, batch_pos))
                        self.total_samples += b

    def find_matching_target_file(self, input_file, target_files):
        input_file_name = os.path.basename(input_file)
        input_file_name_prefix = input_file_name.split(f"_{self.input_gsd_pattern}_{self.bands_pattern}.pt")[0]
        for target_file in target_files:
            target_file_name = os.path.basename(target_file)
            target_file_name_prefix = target_file_name.split(f"_05m_{self.bands_pattern}.pt")[0]
            if input_file_name_prefix == target_file_name_prefix:
                return target_file
        return None
    
    def find_matching_gpkg(self, input_file):
        matching_gpkg = '_'.join(input_file.split('_')[:-2])+"_patches.gpkg"
        print(matching_gpkg)
        return matching_gpkg

    def get_num_samples(self, file_path):
        tensor = torch.load(file_path)
        return tensor.size(0)
        
    def download(self, site_name=None):
        import py7zr
        # We just always download everything: the X4/X8 datasets are not big anyway
        for data in self.urls:
            filename = None
            md5sum = None
            if isinstance(data, str):
                url = data
                if not filename:
                    filename = os.path.basename(url)
            else:
                url = data[0]
                if len(data) > 1:
                    md5sum = data[1]
                if len(data) > 2:
                    filename = data[2]
                else:
                    filename = os.path.basename(url)
            if site_name is not None and site_name not in url:
                continue
            if (self.root, url) in self.already_downloaded_urls:
                continue
            # torchvision.datasets.utils.download_and_extract_archive(url, self.root, filename=filename, md5=md5sum)
            torchvision.datasets.utils.download_url(url, self.root, filename=filename, md5=md5sum)
            with py7zr.SevenZipFile(Path(self.root) / filename, mode='r') as z:
                if not (Path(self.root) / filename).with_suffix('').exists():
                    print('Extracting 7zip archive')
                    z.extractall()
            self.already_downloaded_urls.append((self.root, url))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        input_file, target_file, batch_pos = self.samples[idx]
        input_tensor = torch.load(input_file)[batch_pos]/self.SCALE
        target_tensor = torch.load(target_file)[batch_pos]/self.SCALE
        
        if self.load_geometry:
            geometry = gpd.read_file(self.find_matching_gpkg(input_file), rows=slice(batch_pos, batch_pos+1))
            return input_tensor, target_tensor, (geometry.to_json(), geometry.crs)
        return target_tensor, input_tensor
    
    def getitem_xarray(self, idx):
        assert self.load_geometry, "Cannot use `getitem_xarray()` if `load_geometry` is False, use `load_geometry = True` when instantiating the dataset."
        inputs, targets, geometry = self.__getitem__(idx)
        gdf = gpd.GeoDataFrame.from_features(json.loads(geometry[0]), crs=geometry[1])
        
        minx,miny,maxx,maxy = gdf.total_bounds
        # inputs
        gsd = (maxx-minx) / (inputs.shape[-1])
        xs = np.arange(minx, maxx, gsd)+gsd/2
        ys = np.arange(maxy, miny, -gsd)-gsd/2
        da_input = xr.DataArray(inputs, dims=['band', 'y', 'x'], coords={'band': ['b2','b3','b4','b8'], 'y':ys, 'x':xs })
        da_input = da_input.rio.write_crs(gdf.crs)

        # targets
        gsd = (maxx-minx) / (targets.shape[-1])
        xs = np.arange(minx, maxx, gsd)+gsd/2
        ys = np.arange(maxy, miny, -gsd)-gsd/2
        da_target = xr.DataArray(targets, dims=['band', 'y', 'x'], coords={'band': ['b2','b3','b4','b8'], 'y':ys, 'x':xs })
        da_target = da_target.rio.write_crs(gdf.crs)

        return da_input, da_target

