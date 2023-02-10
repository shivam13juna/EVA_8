'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# class eva_s6(nn.Module):
#     def __init__(self, num_classes=10):
#         super(eva_s6, self).__init__()
#         self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
#         self.depthwise_conv = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64)
#         self.dilated_conv = nn.Conv2d(64, 64, 3, stride=1, padding=2, dilation=2)
#         self.gap = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(64, self.num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.depthwise_conv(x))
#         x = F.relu(self.dilated_conv(x))
#         x = self.gap(x)
#         x = x.view(-1, 64)
#         x = self.fc(x)
#         return x

# n_out = (n_in - k + 2*p)/s + 1
# j_out = j_in * s
# r_out = r_in + (k - 1) * j_in
# j_in = j_out_previous, initially 1



class eva_s6(nn.Module):
    def __init__(self, num_classes=10):
        super(eva_s6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=0, stride=2) # n_in = 32, k = 5, p = 0, s = 2, n_out = 14, j_in = 1, j_out = 2, r_out = 5

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0, stride=2) # n_in = 14, k = 5, p = 0, s = 2, n_out = 5, j_in = 2, j_out = 4, r_out = 13

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=2, groups=32) # n_in = 5, k = 5, p = 0, s = 2, n_out = 1, j_in = 4, j_out = 8, r_out = 29

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Move model to GPU


# Move model to GPU
# model = Net().to(device)




# test()
