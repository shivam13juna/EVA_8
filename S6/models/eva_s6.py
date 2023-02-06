'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class eva_s6(nn.Module):
    def __init__(self, num_classes=10):
        super(eva_s6, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.depthwise_conv = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64)
        self.dilated_conv = nn.Conv2d(64, 64, 3, stride=1, padding=2, dilation=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.depthwise_conv(x))
        x = F.relu(self.dilated_conv(x))
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x
# class eva_s6(nn.Module):
#     def __init__(self, num_classes=10):
#         super(eva_s6, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(256 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = x.view(-1, 256 * 8 * 8)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# Move model to GPU


# Move model to GPU
# model = Net().to(device)




# test()
