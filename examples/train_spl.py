from sen2venus import Sen2VenusSite, Sen2Venus, Sen2VenusURLs
import fire
import os
import sys
sys.path.append('sr-pytorch-lightning')
from models import SRGAN
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
import torch
from torch.utils.data import DataLoader


# 8-band order of the Sen2Venus 'all' subset: b2,b3,b4,b8 then the (upsampled)
# 20m bands re1,re2,re3,re4(=b8a). Earth-search common-name asset keys below
# are listed in the exact same order so the model receives channels as trained.
STAC_BANDS = ["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08"]
# Sentinel-2 reflectance is stored as DN; divide by this to get [0, 1].
SCALE = 10000.0
# SCL classes to drop when masking clouds: 0 no-data, 1 defective, 3 cloud shadow,
# 8/9 cloud medium/high prob, 10 thin cirrus, 11 snow/ice.
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]


def train(device='cpu', datasets_root='.', batch_size=20, num_workers=0, max_epochs=-1,
          experiment_name='sen2venus-srgan', run_name=None, mlflow_tracking_uri=None,
          log_dir='lightning_logs'):
    dataset = Sen2Venus(datasets_root, Sen2VenusURLs().get_sites_list(), subset='all', load_geometry=False, return_type='dict', device=device, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers)
    model = SRGAN(scale_factor=2, channels=8, batch_size=batch_size)
    model.automatic_optimization = False

    # NOTE: TensorBoardLogger MUST stay first in the list. SRGAN.training_step logs
    # images via `self.logger.experiment.add_image(...)`, and `self.logger` resolves to
    # loggers[0]; MLFlowLogger.experiment has no add_image and would crash training.
    # Scalar metrics logged via `log_dict` are fanned out to ALL loggers, so MLflow
    # still receives every metric.
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=mlflow_tracking_uri or f"file:{os.path.abspath('mlruns')}",
        log_model=False,
    )
    mlf_logger.log_hyperparams({
        'scale_factor': 2,
        'channels': 8,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'subset': 'all',
        'augment': True,
        'device': device,
        'max_epochs': max_epochs,
        'g_lr': 1e-4,
        'd_lr': 1e-4,
        'gan_mode': 'wgangp',
    })

    trainer = Trainer(
        accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
        max_epochs=max_epochs,
        log_every_n_steps=100,
        logger=[tb_logger, mlf_logger],
    )
    trainer.fit(model, train_dataloaders=dataloader)


def test(checkpoint, device='cpu', datasets_root='.'):
    raise NotImplementedError()


def predict(checkpoint, device='cpu', aoi=(1.563, 43.675, 1.594, 43.697), toi='2023-07',
            max_cloud_cover=20, output_path=None, save_input_tifs=True, mask_clouds=True,
            harmonize_offset=True, stac_url='https://earth-search.aws.element84.com/v1',
            collection='sentinel-2-c1-l2a'):
    """Super-resolve Sentinel-2 imagery pulled from a public STAC catalog.

    Args:
        checkpoint: path to a trained SRGAN Lightning checkpoint.
        device: torch device ('cpu' or 'cuda').
        aoi: (minx, miny, maxx, maxy) bounding box in lon/lat (EPSG:4326).
        toi: time of interest as a STAC datetime / interval (e.g. '2023-07').
        max_cloud_cover: keep only scenes with eo:cloud_cover below this percentage.
        output_path: output folder/prefix. Defaults to a hash of aoi+toi.
        save_input_tifs: also write the (10m) model input as GeoTIFF.
        mask_clouds: mask invalid SCL classes (sets them to 0 before inference).
        harmonize_offset: subtract the BOA add-offset (baseline 04.00+ scenes carry
            -1000 DN) before scaling, to match the Sen2Venus training distribution.
        stac_url: STAC API root.
        collection: STAC collection id.
    """
    import numpy as np
    import xarray as xr
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import odc.stac
    from pystac_client import Client

    print("Searching STAC catalog...")
    catalog = Client.open(stac_url)
    search = catalog.search(
        collections=[collection],
        bbox=aoi,
        datetime=toi,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())
    if not items:
        raise SystemExit(f"No scenes found for bbox={aoi} datetime={toi} cloud<{max_cloud_cover}%.")
    print(f"Found {len(items)} scene(s).")

    # Resolve the BOA add-offset from the first item's metadata (falls back to -1000,
    # the standard value for processing baseline >= 04.00). Earth-search delivers raw DN.
    offset = 0.0
    if harmonize_offset:
        offset = -1000.0
        try:
            rb = items[0].assets[STAC_BANDS[0]].extra_fields.get("raster:bands")
            if rb and rb[0].get("offset") is not None:
                offset = float(rb[0]["offset"]) * SCALE  # offset is in reflectance units
        except (KeyError, IndexError, TypeError):
            pass
    print(f"Using BOA offset = {offset} DN, scale = {SCALE}.")

    print("Loading imagery (10m, cubic upsampling of 20m bands)...")
    resampling = {b: "cubic" for b in ("rededge1", "rededge2", "rededge3", "nir08")}
    resampling.update({"scl": "nearest", "*": "nearest"})
    ds = odc.stac.load(
        items,
        bands=STAC_BANDS + (["scl"] if mask_clouds else []),
        bbox=aoi,
        resolution=10,
        resampling=resampling,
        groupby="solar_day",
        chunks={},
        dtype="float32",
    )

    da = ds[STAC_BANDS].to_array("band").transpose("time", "band", "y", "x")
    da = (da + offset) / SCALE
    if mask_clouds:
        valid = ~ds["scl"].isin(SCL_INVALID)
        da = da.where(valid)
    da = da.clip(0, 1).compute()
    print(f"Loaded cube: {dict(da.sizes)}")

    print("Loading model...")
    model = SRGAN.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()

    # Build a corner-aligned output grid at native_res / scale_factor. We do NOT use
    # np.linspace(x[0], x[-1], n*2): that spans center-to-center and offsets/under-covers
    # the upsampled grid by half a coarse pixel.
    sf = 2
    res = float(abs(da.x[1] - da.x[0]))  # native pixel size (10m)
    new_res = res / sf
    nx, ny = da.sizes["x"] * sf, da.sizes["y"] * sf
    new_x = float(da.x[0]) - res / 2 + new_res / 2 + np.arange(nx) * new_res
    new_y = float(da.y[0]) + res / 2 - new_res / 2 - np.arange(ny) * new_res

    if output_path is None:
        output_path = '_'.join(['-'.join(str(c) for c in aoi), toi])
    os.makedirs(output_path, exist_ok=True)
    base = os.path.basename(output_path.rstrip('/')) or 'sr'

    print("Predicting per date...")
    for t in da.time.values:
        date = str(t)[:10]
        da_t = da.sel(time=t)
        arr = np.nan_to_num(da_t.values, nan=0.0).astype("float32")
        if not np.any(arr):
            print(f"  {date}: fully masked/empty, skipping.")
            continue
        x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, band, y, x)
        with torch.no_grad():
            y = model(x).squeeze(0).cpu().numpy()  # (band, 2y, 2x)

        out = xr.DataArray(
            y, dims=["band", "y", "x"],
            coords={"band": STAC_BANDS, "y": new_y, "x": new_x},
        ).rio.write_crs(da.rio.crs)
        out.rio.to_raster(os.path.join(output_path, f"{base}_{date}_sr.tif"))
        if save_input_tifs:
            da_t.rio.write_crs(da.rio.crs).rio.to_raster(
                os.path.join(output_path, f"{base}_{date}_input.tif"))
        print(f"  {date}: wrote {base}_{date}_sr.tif")

    print("Done.")


if __name__ == '__main__':
    fire.Fire({"train": train, "test": test, "predict": predict})
