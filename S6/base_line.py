import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

seed_everything(7)

from torchmetrics import Accuracy

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


# ### CIFAR10 Data Module
#
# Import the existing data module from `bolts` and modify the train and test transforms.


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)


# ### Model Architecture
# Modify the pre-existing Resnet architecture from TorchVision. The pre-existing architecture is based on ImageNet
# images (224x224) as input. So we need to modify it for CIFAR10 images (32x32).

class PrintShape(nn.Module):
    def __init__(self, text=None):
        super(PrintShape, self).__init__()
        self.text = text

    def forward(self, x):
        # print("For the: ", self.text, " the shape is: ", x.shape)
        return x


class eva_s6(nn.Module):
    def __init__(self, num_classes=10):
        super(eva_s6, self).__init__()
        
        # n_out = (n_in - k + 2*p)/s + 1
        # j_out = j_in * s
        # r_out = r_in + (k - 1) * j_in
        # j_in = j_out_previous, initially 1

        # Output size = (Input size + 2 * padding - dilation * (kernel size - 1) - 1) / stride + 1

        # n_out = (n_in + 2* p - d*(k-1) - 1)/s + 1

        # so if d == 1 then n_out = (n_in + 2*p - k)/s + 1
        # so if d == 2 then n_out = (n_in + 2*p - 2k + 1)/s + 1
        # so if d == 3 then n_out = (n_in + 2*p - 3k + 2)/s + 1

        # Input Block
        self.conv1_block = nn.Sequential(
            PrintShape("conv1_block_input"),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False), 
            # n_in = 32, k = 3, p = 1, n_out = 32, j_in = 1, j_out = 1, r_out = 5; 32 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),

            PrintShape("conv1_block_c1_output"),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False),
            # n_in = 32, k = 5, p = 1, n_out = 32, j_in = 1, j_out = 1, r_out =  9; 32 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),

            PrintShape("conv1_block_c2_output"),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2, bias=False),
            # n_in = 32, k = 5, p = 1, n_out = 32, j_in = 1, j_out = 1, r_out = 13; 32 x 32 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05),

            PrintShape("conv1_block_c3_output")
            )


# so if d == 2 then n_out = (n_in + 2*p - 2k + 1)/s + 1
# 32 + 2 * 0 
        
        self.conv2_block = nn.Sequential(

            PrintShape("conv2_block_input"),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2, bias=False),
            # n_in = 32, k = 5, p = 1, n_out = 32, j_in = 1, j_out = 1, r_out = 17; 32 x 32 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05),

            PrintShape("conv2_block_c1_output"),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation = 2, groups=64, bias=False),
            # n_in = 32, k = 5, p = 0, n_out = 28, j_in = 1, j_out = 1, r_out = 21; 28 x 28 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),

            PrintShape("conv2_block_c2_output"),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, dilation=2, bias=False),
            # n_in = 28, k = 5, p = 0, n_out = 24, j_in = 1, j_out = 1, r_out = 25; 26 x 26 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),

            PrintShape("conv2_block_c3_output")
            )

# so if d == 3 then n_out = (n_in + 2*p - 3k + 2)/s + 1
        self.conv3_block = nn.Sequential(

            PrintShape("conv3_block_input"),


            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, dilation=3, bias=False),
            # n_in = 24, k = 6, p = 0, n_out = 19, j_in = 1, j_out = 1, r_out = 30; 20 x 20 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),

            PrintShape("conv3_block_c1_output"),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, dilation=3, bias=False),
            # n_in = 18, k = 6, p = 0, n_out = 14, j_in = 1, j_out = 1, r_out = 35; 16 x 16 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),

            PrintShape("conv3_block_c2_output"),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, dilation=3, bias=False),
            # n_in = 12, k = 6, p = 0, n_out = 9, j_in = 1, j_out = 1, r_out = 40; 11 x 11 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),

            PrintShape("conv3_block_c3_output"),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, dilation=3, bias=False),
            # n_in = 6, k = 6, p = 0, n_out = 4, j_in = 1, j_out = 1, r_out = 45; 6 x 6 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),

            PrintShape("conv3_block_c4_output")
            )

            # Use global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) # output_size = 1

        self.fc1 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.fc1(x)


        
        return x
# class eva_s6(nn.Module):
#     def __init__(self, num_classes=10):
#         super(eva_s6, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1, stride=2) # n_in = 32, k = 5, p = 0, s = 2, n_out = 14, j_in = 1, j_out = 2, r_out = 5

#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=2, dilation=2) # n_in = 14, k = 5, p = 0, s = 2, n_out = 5, j_in = 2, j_out = 4, r_out = 13

#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0, stride=2, groups=16) # n_in = 5, k = 5, p = 0, s = 2, n_out = 1, j_in = 4, j_out = 8, r_out = 29

#         # Use global pooling
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.inp_fc1 = 128
#         self.fc1 = nn.Linear(in_features=self.inp_fc1, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=32)
#         self.fc3 = nn.Linear(in_features=32, out_features=10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         print("Shape after conv1: ", x.shape)
#         x = F.relu(self.conv2(x))
#         print("Shape after conv2: ", x.shape)
#         x = F.relu(self.conv3(x))
#         print("Shape after conv3: ", x.shape)
#         x = self.global_pool(x)
#         print("Shape after global_pool: ", x.shape)
#         # print shape of x
#         print(x.shape)
#         x = x.view(-1, self.inp_fc1)
#         # x = self.fc1(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# print model summary using torchsummary
eva = eva_s6()
eva = eva.to("cuda")
from torchsummary import summary

summary(eva, input_size=(3, 32, 32))


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = eva_s6()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


# ### Train model


model = LitResnet(lr=0.05)

trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)
