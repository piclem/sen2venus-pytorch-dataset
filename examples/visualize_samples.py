from sen2venus import Sen2VenusSite
import matplotlib.pyplot as plt
import fire

def visualize_samples(idx=0, subset='all'):
    dataset = Sen2VenusSite('./', 'SUDOUE-4', load_geometry=True, subset='all')
    input, target = dataset.getitem_xarray(idx)
    input.plot.imshow(col='band', vmin=0, vmax=0.6, cmap='gray')
    target.plot.imshow(col='band', vmin=0, vmax=0.6, cmap='gray')
    plt.show()

if __name__ == '__main__':
    fire.Fire(visualize_samples)
    