
# # Importing Libraries


import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)

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
from torchmetrics import Accuracy
from hydra import compose, initialize
from torch import nn, optim, utils, Tensor
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split, DataLoader, TensorDataset


# shutil.copy("configs/config.yaml", "notebooks/config.yaml")
# with initialize(version_base=None, config_path=""):
#     config = compose(config_name="config.yaml")



# Creating Data-Loader for MNIST

mnist_mean = (0.1307,)
mnist_std = (0.3081,)


class custom_mnist_dataset(torchvision.datasets.MNIST):
    def __init__(
        self,
        root="data/",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    ):
        super().__init__(root, train, transform, target_transform, download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # print("Type of img: ", type(img))
        img = np.array(img)
        # img = img.reshape(1, 28, 28)
        # convert image to PIL image
        img = Image.fromarray(img)
        # Add a channel dimension to the image using torch
        # img = torch.unsqueeze(img, 0)
        # # convert to float
        # img = img.float()

        if self.transform is not None:

            transformed = self.transform(img)
            img = transformed

        # img = torch.from_numpy(img)
        ano = np.random.randint(10)
        return (img, ano), target


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mnist_mean, mnist_std),
            ]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            mnist_full = custom_mnist_dataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = custom_mnist_dataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class custom_view(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MNISTTrainingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Model 1 takes image as the input and outputs 10 classes
        # Input size of image is 28x28x1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )  # after conv1 output size is 28x28x32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # after conv2 output size is 28x28x64
        self.max_pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # after max_pool1 output size is 14x14x64

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3
        )  # after conv3 output size is 12x12x128
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3
        )  # after conv4 output size is 10x10x256
        self.max_pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # after max_pool2 output size is 5x5x256
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3
        )  # after conv5 output size is 3x3x512
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3
        )  # after conv6 output size is 1x1x1024

        self.view = custom_view()  # after view output size is 1024

        self.fc1 = nn.Linear(
            in_features=1024, out_features=256
        )  # after fc1 output size is 10
        self.fc2 = nn.Linear(
            in_features=256, out_features=128
        )  # after fc2 output size is 1
        self.fc3 = nn.Linear(
            in_features=128, out_features=10
        )  # after fc3 output size is 10
        self.cv_model = nn.Sequential(
            self.conv1,
            self.conv2,
            self.max_pool1,
            self.conv3,
            self.conv4,
            self.max_pool2,
            self.conv5,
            self.conv6,
            self.view,
            self.fc1,
            self.fc2,
            self.fc3,
        )

        self.cv_pass = nn.Sequential(
            self.conv1,
            self.conv2,
            self.max_pool1,
            self.conv3,
            self.conv4,
            self.max_pool2,
            self.conv5,
            self.conv6,
            self.view,
            self.fc1,
            self.fc2,
        )

        # Model 2 takes one-hot encoded number (in the range (0 to 9)) as the input and outputs the sum of the label from the image and one-hot encoded number

        self.model2_fc1 = nn.Linear(in_features=1, out_features=128)
        # self.model2_fc2 = nn.Linear(in_features=128, out_features=256)
        self.final_layer = nn.Linear(in_features=256, out_features=1)
        self.accuracy = Accuracy()

        # self.num_model = nn.Sequential(self.model2_fc1, self.model2_fc2, self.model2_fc3)

        # stack the layers of self.fc2 and self.model2_fc4

        # self.final_layer = nn.Linear(in_features=256, out_features=2)

    # Calcualate custom loss, sparse categorical cross entropy for sth1 and mean squared error for sth4

    def custom_loss(self, sth1, sth4, x, y):
        loss1 = F.cross_entropy(sth1, y)
        loss2 = F.mse_loss(sth4, y.unsqueeze(1) + x)

        return loss1 + loss2

    def training_step(self, batch, batch_idx):
        x, y = batch
        inp1 = x[0].to(self.device)
        inp2 = x[1].unsqueeze(1).type(torch.FloatTensor).to(self.device)

        y = y.to(self.device)

        sth1 = self.cv_model(inp1)
        sth2 = self.model2_fc1(inp2)

        tmp = self.cv_pass(inp1)
        sth3 = torch.cat((tmp, sth2), 1)
        sth4 = self.final_layer(sth3)

        # calculate loss
        loss = self.custom_loss(sth1, sth4, inp2, y)

        # print("Shape of inp2 + y: ", (inp2 + y.unsqueeze(1)).shape, " shape of inp2: ", inp2.shape, " shape of y: ", y.unsqueeze(1).shape)
        # sys.exit()

        # log loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inp1 = x[0].to(self.device)
        inp2 = x[1].unsqueeze(1).type(torch.FloatTensor).to(self.device)

        y = y.to(self.device)

        sth1 = self.cv_model(inp1)
        sth2 = self.model2_fc1(inp2)

        tmp = self.cv_pass(inp1)
        sth3 = torch.cat((tmp, sth2), 1)
        sth4 = self.final_layer(sth3)

        # calculate loss
        loss = self.custom_loss(sth1, sth4, inp2, y)

        # log loss
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        sth1 = self.cv_model(x[0])
        sth2 = self.model2_fc1(x[1])
        sth3 = torch.concat(sth1, sth2)
        sth4 = self.final_layer(sth3)

        return F.log_softmax(sth1, dim=1), sth4

    def test_step(self, batch, batch_idx):
        x, y = batch
        inp1 = x[0].to(self.device)
        inp2 = x[1].unsqueeze(1).type(torch.FloatTensor).to(self.device)

        y = y.to(self.device)

        sth1 = self.cv_model(inp1)
        sth1 = F.log_softmax(sth1, dim=1)
        sth2 = self.model2_fc1(inp2)

        tmp = self.cv_pass(inp1)
        sth3 = torch.cat((tmp, sth2), 1)
        sth4 = self.final_layer(sth3)

        # calculate acc
        acc1 = self.accuracy(sth1, y)

        # Cast sth4 and y.unsqueeze(1) + inp2 to int
        sth4 = sth4.squeeze().type(torch.IntTensor)
        ano = y.unsqueeze(1) + inp2
        ano = ano.squeeze().type(torch.IntTensor)

        print("Shape of sth4: ", sth4.shape, " shape of ano: ", ano.shape)

        acc2 = self.accuracy(sth4, ano)

        # log acc
        self.log("test_acc1", acc1)
        self.log("test_acc2", acc2)


data_module = MnistDataModule(batch_size=32)
model = MNISTTrainingModule()

trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(model, data_module)


trainer.test(model, datamodule=data_module)


## Questions
# Your code MUST be:
#     well documented (via readme file on GitHub and comments in the code)
#     must mention the data representation
#     must mention your data generation strategy (basically the class/method you are using for random number generation)
#     must mention how you have combined the two inputs (basically which layer you are combining)
#     must mention how you are evaluating your results 
#     must mention "what" results you finally got and how did you evaluate your results
#     must mention what loss function you picked and why!
#     training MUST happen on the GPU
#     Accuracy is not really important for the SUM

# 1. What is data representation? 
# I didn't use a one-hot encoding for the random number, instead in the data module for MNIST, I modified it to include a random number, so this was my input (img, ano), target. Where ano is the random number.

# 2. What is data generation strategy?
# In the pytorch lightning data module, I added a custom dataset module, and in the __getitem__ method, I added ano = torch.randint(0, 10, (1,)) to generate a random number between 0 and 10. and returned (img, ano), target in the __getitem__ method.

# 3. How did I combine the two inputs?
# In the second last layer of the convolutional network(MNIST prediction label) I've a fully connected layer with in-features 256 and out-features as 128, I concatenated the output of this layer with the output of the fully connected layer of the with input as random number, and then passed it through a final fully connected layer with in-features 256 and out-features as 2. So the output of this layer is a 2D tensor with 2 features, which is the predicted label and the sum of the random number and the predicted label.

# 4. How did I evaluate my results?
# I used the accuracy metric to evaluate my results. I used the accuracy metric for the MNIST prediction label and the sum of the random number and the predicted label. For latter I converted the predicted sum to int and then used the accuracy metric.

# 5. What results did I get?
# I got an accuracy of 0.98 for the MNIST prediction label and 0.40 for the sum of the random number and the predicted label.

# 6. What loss function did I pick and why?
# I used a custom loss function, in which the first value was used to compute the cross entropy loss for the MNIST prediction label and the mean squared error loss for the sum of the random number and the predicted label. I used cross-entropy loss for the MNIST prediction label because it is a classification problem and mean squared error loss for the sum of the random number and the predicted label because it is a regression problem.

# 7. Training on GPU
# I used the GPU for training.

