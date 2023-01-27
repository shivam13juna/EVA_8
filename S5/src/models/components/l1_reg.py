import torch
from torch import nn
from src.models.components.bn_model import ConvNet


class L1RegularizedConv2D(nn.Module):
    def __init__(self, model, weight_decay=0.0001):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def regularization(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if "conv2d" in name.lower():
                reg_loss += torch.norm(param, 1)
        return reg_loss * self.weight_decay


if __name__ == "__main__":
    _ = L1RegularizedConv2D(ConvNet())
