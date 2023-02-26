## Clone Deep Learning Utils Repo
# !git clone https://github.com/shivam13juna/eva8_utils.git


import pyrootutils
import sys

root = pyrootutils.setup_root(
    search_from=sys.path[0],
    pythonpath=True,
    cwd=True,
)

import torch
from core_utils import main
from core_utils.utils import data_handling, train, test, gradcam, helpers, augmentation
from core_utils.models import s9_attention
from pprint import pprint
from torch_lr_finder import LRFinder


import timm
import urllib
import torch
import os
import numpy as np

import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt


# ## Import Config


config_file_path = "core_utils/config/s8_config.yaml"

config = helpers.load_config_variables(config_file_path)


# ## Perform GPU Check; Create "Device"


device, cuda = helpers.gpu_check(seed_value=1)
pprint(device)


# ## Download Dataset, Create Data Loaders


mean, std = data_handling.return_dataset_statistics()
trainloader, testloader = main.create_dataloaders(
    mean,
    std,
    cuda,
    config,
    gpu_batch_size=512,
    augment_func="s8_albumentation_augmentation",
)


# ## Model Summary


from torchsummary import summary

model = s9_attention.Att_Model().to(device)
summary(model, input_size=(3, 32, 32))


# # make trainloader iterator
# import torchvision

# # create train_loader from torchvision datasets cifar10
# transform_train = T.Compose([
#      T.ToTensor(),
#     T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),

# ])

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# new_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# criterion = torch.nn.CrossEntropyLoss()
# lr_finder = LRFinder(model, optimizer, criterion, device=device)
# lr_finder.range_test(new_trainloader, end_lr=10, num_iter=200)
# lr_finder.plot()


# ## Trigger Training


train_acc, train_losses, test_acc, test_losses, lrs = main.start_training(
    model,
    device,
    trainloader,
    testloader,
    config,
    optimizer_name="Adam",
    # scheduler_name="ReduceLROnPlateau",
    scheduler_name="OneCycle",
    criterion_name="CrossEntropyLoss",
    epochs=24,
)


# ## Plot Metrics


helpers.plot_metrics(train_acc, train_losses, test_acc, test_losses)
