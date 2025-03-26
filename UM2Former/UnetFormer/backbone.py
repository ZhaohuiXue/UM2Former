import torch.nn as nn


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=24):
        super(ResNet18, self).__init__()
        self.dim = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, self.dim[0], 2, stride=1)
        self.layer2 = self.make_layer(64, self.dim[1], 2, stride=2)
        self.layer3 = self.make_layer(self.dim[1], self.dim[2], 2, stride=2)
        self.layer4 = self.make_layer(self.dim[2], self.dim[3], 2, stride=2)

    def feature_info(self):
        return self.dim

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        layer.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layer.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        res1 = self.layer1(out)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        return res1, res2, res3, res4


class ResNet34(nn.Module):
    def __init__(self, num_classes=24):
        super(ResNet34, self).__init__()
        self.dim = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, self.dim[0], 2, stride=1)
        self.layer2 = self.make_layer(64, self.dim[1], 4, stride=2)
        self.layer3 = self.make_layer(self.dim[1], self.dim[2], 6, stride=2)
        self.layer4 = self.make_layer(self.dim[2], self.dim[3], 3, stride=2)

    def feature_info(self):
        return self.dim

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        layer.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layer.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        res1 = self.layer1(out)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        return res1, res2, res3, res4