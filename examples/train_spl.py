from sen2venus import Sen2VenusSite, Sen2Venus, Sen2VenusURLs
import matplotlib.pyplot as plt
import fire
import sys
sys.path.append('sr-pytorch-lightning')
from models import SRGAN
from lightning import Trainer
from torch.utils.data import DataLoader

def train(device='cpu', datasets_root='.'):
    dataset = Sen2Venus(datasets_root, Sen2VenusURLs().get_sites_list(), subset='all', load_geometry=False, return_type='dict', device=device, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    model = SRGAN(scale_factor=2, channels=8)
    model.automatic_optimization = False
    trainer = Trainer(log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=dataloader)

if __name__ == '__main__':
    fire.Fire(train)
    
