from sen2venus import Sen2VenusSite, Sen2Venus, Sen2VenusURLs
import matplotlib.pyplot as plt
import fire
import sys
sys.path.append('sr-pytorch-lightning')
from models import SRGAN
from lightning import Trainer
import torch
from torch.utils.data import DataLoader

def train(device='cpu', datasets_root='.'):
    dataset = Sen2Venus(datasets_root, Sen2VenusURLs().get_sites_list(), subset='all', load_geometry=False, return_type='dict', device=device, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    model = SRGAN(scale_factor=2, channels=8)
    model.automatic_optimization = False
    trainer = Trainer(log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=dataloader)

def test(checkpoint, device='cpu', datasets_root='.'):
    raise NotImplementedError()

def predict(checkpoint, device='cpu', aoi=(1.563, 43.675, 1.594, 43.697), toi='2023-07', output_path=None, save_input_tifs=True):
    from earthdaily.earthdatastore import Auth
    import numpy as np
    import xarray as xr
    import rioxarray as rxr
    import os

    tmp_file_hash = '_'.join(['-'.join([str(c) for c in aoi]), toi])
    tmp_input_file = tmp_file_hash + '_input.zarr'
    print("Downloading data...")
    if not os.path.exists(tmp_input_file):
        client = Auth()
        data = client.datacube('sentinel-2-l2a', assets=["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08"], datetime=toi, bbox=aoi, clear_cover=100, mask_with='scl', resampling='cubic')
        print(f"Found {len(data.coords['time'])} clear dates: {data.coords['time'].values}")
        data = data.to_array('band').compute()
        data.to_zarr(tmp_input_file)
    else:
        data = xr.open_zarr(tmp_input_file).to_array('v').isel(v=0) # force DataArray 

    # convert to torch tensor
    data = data.transpose('time', 'band', 'y', 'x')
    x = torch.from_numpy(np.nan_to_num(data.values, nan=0)).to('cuda') # time is batch dimension, no need to unsqueeze
    print("Loading model...")
    model = SRGAN.load_from_checkpoint(checkpoint)
    model.automatic_optimization = False
    print("Predict...")
    with torch.no_grad():
        y = model(x).cpu()
    print("Save output...")

    # TODO: that's probably dirty coordinates definition (corner vs center etc.) => check that
    new_x = np.linspace(data.x[0], data.x[-1], data.sizes["x"] * 2)
    new_y = np.linspace(data.y[0], data.y[-1], data.sizes["y"] * 2)
    output_data = data.interp(x=new_x, y=new_y)
    output_data.values = y
    if output_path is None:
        output_path = tmp_file_hash
    os.makedirs(output_path, exist_ok=True)
    for t in output_data.coords['time'].values:
        output_data.sel(time=t).rio.to_raster(os.path.join(output_path, output_path+f'_{str(t)[:10]}_sr.tif'))
        data.sel(time=t).rio.to_raster(os.path.join(output_path, output_path+f'_{str(t)[:10]}_input.tif'))

    print("Done.")
    
if __name__ == '__main__':
    fire.Fire({"train":train, "test":test, "predict": predict})
    
