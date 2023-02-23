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
from core_utils.models import resnet
from pprint import pprint


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


config_file_path = "core_utils/config/config.yaml"

config = helpers.load_config_variables(config_file_path)


# ## Perform GPU Check; Create "Device"


device, cuda = helpers.gpu_check(seed_value=1)
pprint(device)


# ## Download Dataset, Create Data Loaders


mean, std = data_handling.return_dataset_statistics()
trainloader, testloader = main.create_dataloaders(
    mean, std, cuda, config, augment_func="albumentation_augmentation"
)


# ## Model Summary


from torchsummary import summary

model = resnet.ResNet18("layer").to(device)
summary(model, input_size=(3, 32, 32))


# ## Trigger Training


train_acc, train_losses, test_acc, test_losses, lrs = main.start_training(
    model,
    device,
    trainloader,
    testloader,
    config,
    optimizer_name="Adam",
    scheduler_name="ReduceLROnPlateau",
    criterion_name="CrossEntropyLoss",
    lambda_l1=0,
    epochs=20,
)


# ## Plot Metrics


helpers.plot_metrics(train_acc, train_losses, test_acc, test_losses)


# ## Misclassified Images


misclassified_images = helpers.wrong_predictions(model, testloader, device)
helpers.plot_misclassified(misclassified_images, mean, std, 20)


# ## Gradcam


# cifar 10 labels in a dictionary

cifar10_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


from core_utils.utils import gradcam

gradcam.plot_gradcam_images(
    model,
    [model.layer4[-1]],
    misclassified_images[:10],
    list(range(10)),
    cifar10_labels,
)
