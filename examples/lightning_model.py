import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from sen2venus import Sen2Venus


# TODO: add use_bn
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # TODO: implement without bn if relevant
        #out = self.relu(self.conv1(x))
        #out = self.conv2(out)

        if self.stride != 1 or x.shape[1] != out.shape[1]:
            residual = F.avg_pool2d(residual, self.stride)
            residual = torch.cat((residual, torch.zeros_like(residual)), dim=1)

        # TODO: check before/after relu for residual addition
        out += residual
        out = self.relu(out)
        return out
    
# TODO: add use_bn
# TODO: add predict_residual=False (=> u = 0 if ... else x)
class SuperResolutionModel(pl.LightningModule):

    def __init__(self, scale=2, in_channels=4, n_res_blocks=3):
        super(SuperResolutionModel, self).__init__()
        self.scale = scale
        self.residual_scale = 0.1  # scale the residual added to the upscaled input image
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, padding=2)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for i in range(n_res_blocks)],
        )
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=in_channels*scale**2, kernel_size=3, padding=1)
        self.upscale = nn.PixelShuffle(scale)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        # upscale the input
        # 'bicubic' is a better approximation but 'nearest' reflects default 
        #   upscaling modes in libraries such as stackstac
        up = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        x = self.relu(self.conv_in(x))
        x = self.residual_blocks(x)
        x = self.upscale(self.conv_out(x))
        x = up + x * self.residual_scale
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        if batch_idx % 100 == 0:
            self.logger.experiment.add_image('input', x[0,(2,1,0),:,:]*3, self.global_step)
            self.logger.experiment.add_image('target', y[0,(2,1,0),:,:]*3, self.global_step)
            self.logger.experiment.add_image('prediction', y_hat[0,(2,1,0),:,:]*3, self.global_step)           

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Fonction de chargement des données (pour l'exemple, nous utilisons CIFAR-10)
def load_data(batch_size=1, num_workers=1):
    train_dataset = Sen2Venus('./')
    # train_dataset.download('FGMANAUS')
    train_dataset.download('SUDOUE-4')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Train dataset ready")
    return train_loader

def main(batch_size=1, num_workers=1, max_epochs=100):
    model = SuperResolutionModel()
    data_loader = load_data(batch_size=batch_size, num_workers=num_workers)
    logger = TensorBoardLogger('tb_logs', name="Sen2Venus SR Lightning")
    trainer = pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu', max_epochs=max_epochs, logger=logger)
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
