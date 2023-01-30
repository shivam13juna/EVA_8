from torch import nn
import sys
# import torchvision.transforms.functional as F
import torch.nn.functional as F
import joblib


class ConvNet(nn.Module):
    def __init__(self, norm_layer=None):
        super().__init__()

        # check if norm_layer is intance of nn.BatchNorm2d, GroupNorm, LayerNorm, etc
        # if norm_layer.func.__name__ == 'BatchNorm2d':
        #     pass
        # elif norm_layer.func.__name__ == 'GroupNorm':
        #     norm_layer = norm_layer(1, )
        
        # print("what is: ", norm_layer.func, repr(norm_layer.func), isinstance(norm_layer.func, nn.BatchNorm2d), type(norm_layer.func))
        # joblib.dump(norm_layer.func, "tmp/norm_layer.pkl")
        # sys.exit()

        self.convblock1 = nn.Sequential(  # 28x28 > 28x28 | RF 3 | jout=1
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            norm_layer(1, 10),
            nn.Dropout2d(0.05),
        )
        self.convblock2 = nn.Sequential(  # 28x28 > 28x28 | RF 5 | jout=1
            nn.Conv2d(10, 16, 3, padding=1),
            nn.ReLU(),
            norm_layer(1, 16),
            nn.Dropout2d(0.05),
        )
        # TRANSITIONAL BLOCK #1
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 > 14x14 | RF 6 | jout=1
        self.convblock3 = nn.Sequential(  # 14x14 > 14x14 | RF 6 | jout=2
            nn.Conv2d(16, 8, 1), nn.ReLU(), norm_layer(1, 8), nn.Dropout2d(0.05)
        )

        # CONVOLUTION BLOCK #2
        self.convblock4 = nn.Sequential(  # 14x14 > 14x14 | RF 10 | jout=2
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            norm_layer(1, 16),
            nn.Dropout2d(0.05),
        )

        self.convblock5 = nn.Sequential(  # 14x14 > 14x14 | RF 14 | jout=2
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            norm_layer(1, 16),
            nn.Dropout2d(0.05),
        )
        self.convblock6 = nn.Sequential(  # 14x14 > 14x14 | RF 18 | jout=2
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            norm_layer(1, 16),
            nn.Dropout2d(0.05),
        )
        # TRANSITIONAL BLOCK #2
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 > 7x7 | RF 20 | jout=2
        self.convblock7 = nn.Sequential(  # 7x7 > 7x7 | RF 20 | jout=3
            nn.Conv2d(16, 8, 1), nn.ReLU(), norm_layer(1, 8), nn.Dropout2d(0.05)
        )
        # CONVOLUTIONAL BLOCK #3
        self.convblock8 = nn.Sequential(  # 7x7 > 5x5 | RF 26 | jout=3
            nn.Conv2d(8, 10, 3), nn.ReLU(), norm_layer(1, 10), nn.Dropout2d(0.05)
        )
        self.convblock9 = nn.Sequential(  # 5x5 > 3x3 | RF 32 | jout=3
            nn.Conv2d(10, 16, 3), nn.ReLU(), norm_layer(1, 16), nn.Dropout2d(0.05)
        )
        self.convblock10 = nn.Sequential(  # 3x3 > 3x3 | RF 38 | jout=3
            nn.Conv2d(16, 10, 1)
            # nn.ReLU(),
            # norm_layer(1, 10),
            # nn.Dropout2d(0.1)
        )
        self.gap = nn.Sequential(  # 3x3 > 1x1 | RF 42 | jout=3
            nn.AvgPool2d(kernel_size=3)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)


if __name__ == "__main__":
    _ = ConvNet()
