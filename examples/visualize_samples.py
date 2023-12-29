from sen2venus import Sen2Venus
import matplotlib.pyplot as plt
import fire

def visualize_samples(idx=0):
    Sen2Venus('./').download('SUDOUE-4')
    dataset = Sen2Venus('./', load_geometry=True, subset='rededge')
    input, target = dataset.getitem_xarray(idx)
    input.plot.imshow(col='band')
    target.plot.imshow(col='band')
    plt.show()

if __name__ == '__main__':
    fire.Fire(visualize_samples)
    