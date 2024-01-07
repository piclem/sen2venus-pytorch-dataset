from sen2venus import Sen2VenusSite
import matplotlib.pyplot as plt
import fire
import sys
sys.path.append('sr-pytorch-lightning')
from models import SRGAN
from lightning import Trainer
from torch.utils.data import DataLoader

def train(device='cpu'):
    dataset = Sen2VenusSite('./', 'SUDOUE-4', load_geometry=False, subset='all', return_type='dict', device=device, augment=True)
    dataloader = DataLoader(dataset, drop_last=True)
    model = SRGAN(scale_factor=2, channels=8)
    model.automatic_optimization = False
    trainer = Trainer(log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=dataloader)

if __name__ == '__main__':
    fire.Fire(train)
    