import torch
import torch.nn as nn
from kornia.augmentation import RandomMotionBlur, RandomHorizontalFlip, RandomPlanckianJitter, ColorJitter, RandomChannelShuffle, RandomGaussianNoise


class DataAug(nn.Module):
     def __init__(self, probability=0.15, geometric=True):
        super().__init__()
        self.augment_transforms = nn.Sequential(
                RandomMotionBlur(kernel_size=(3,12), angle=(0, 180), direction=(-1,1), p=probability),
                ColorJitter(0.1, 0.1, 0.1, 0.1, p=probability),
                RandomPlanckianJitter(mode='CIED', p=probability),
                RandomGaussianNoise(0, 0.002, p=probability)
                )

     @torch.no_grad()  # disable gradients for effiency
     def forward(self, x, masks=None, depth=None, planes=None):
        transformed = self.augment_transforms(x)
        return transformed
            

