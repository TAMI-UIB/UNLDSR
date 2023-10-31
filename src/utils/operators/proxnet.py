
import torch
from torch import nn
class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class ResBlock(nn.Module):
    def __init__(self, features,  kernel_size):
        super(ResBlock, self).__init__()
        layers = list()
        layers.append(ConvRelu(in_channels=features, out_channels=features,kernel_size=kernel_size))
        layers.append(ConvRelu(in_channels=features, out_channels=features, kernel_size=kernel_size))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
        self.cnn=nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.cnn(x)+x)

class ProxNet(nn.Module):
    def __init__(self, iter, channels, features, kernel_size):
        super(ProxNet, self).__init__()
        layers = []
        layers.append(ConvRelu(in_channels=channels, out_channels=features, kernel_size=kernel_size))
        for _ in range (iter):
            layers.append(ResBlock(features=features, kernel_size=kernel_size))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2))
        self.ResNet = nn.Sequential(*layers)
    def forward(self, x):
        return self.ResNet(x)+x