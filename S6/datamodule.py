import timm
import torch
import shutil
import numpy as np
import torchvision
import seaborn as sns
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms


from PIL import Image
from omegaconf import OmegaConf
from torchvision import datasets
from hydra import compose, initialize
from torch import nn, optim, utils, Tensor
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split, DataLoader, TensorDataset



# import accuracy from torchmetrics
from torchmetrics import Accuracy


cifar_mean = [0.49139968, 0.48215841, 0.44653091]
cifar_std =  [0.24703223, 0.24348513, 0.26158784]


         

class cifar_dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="data/", train=True, download=False, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

        self.transofrm = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
        


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data/', batch_size:int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # use albumentation library and apply:
        #     horizontal flip
        #     shiftScaleRotate
        #     coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
        self.transform_train = A.Compose(
            

                            [
                                A.Resize(32, 32),
                                A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=cifar_mean, mask_fill_value=None),
                            A.Normalize(
                                cifar_mean, cifar_std
                            ),
                            ToTensorV2()
                        ]
                    )



        self.transform_test = A.Compose(
             
            [  A.Resize(32, 32), 
                A.Normalize(
                    cifar_mean, cifar_std
                ),
                ToTensorV2(),
            ]
        )  

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):

        if stage is None:
            self.train_full_set = cifar_dataset(self.data_dir, train=True, download=False, transform=self.transform_train)

            self.cifar_train, self.cifar_val = random_split(self.train_full_set, [45000, 5000])

            self.cifar_test = cifar_dataset(self.data_dir, train=False, download=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
 


