from sen2venus import Sen2VenusV2
import fire


def test_v2(site='SUDOUE-4', subset='all', include_swir=False, idx=0):
    """Load one Sen2Venµs v2.0 sample from the tacofoundation mirror and print shapes.

    Requires the v2 extra: ``pip install 'sen2venus-pytorch-dataset[v2]'``.
    """
    ds = Sen2VenusV2(site_names=site, subset=subset, include_swir=include_swir)
    print(f'{len(ds)} samples for site {site!r}')
    print('lr_bands:', ds.lr_bands)
    print('hr_bands:', ds.hr_bands)
    lr, hr = ds[idx]
    print('lr:', tuple(lr.shape), lr.dtype)
    print('hr:', tuple(hr.shape), hr.dtype)


if __name__ == '__main__':
    fire.Fire(test_v2)
