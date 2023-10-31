import math
from torch import nn

from src.utils.operators.gaussian_blur import GaussianSmoothing, LearnedGaussianSmoothing
class LearnedDownsampling(nn.Module):
    def __init__(self, config_data):
        super(LearnedDownsampling, self).__init__()
        blur = config_data['blur_sd']
        sampling = config_data['sampling']
        channels = config_data['channels']
        gausssina_kernel_size = math.ceil(4*blur)+1
        self.conv = LearnedGaussianSmoothing(channels=channels, kernel_size=gausssina_kernel_size, sigma=blur)
        upsamp_kernel_size = sampling+1 if sampling % 2 == 0 else sampling+2
        self.down_conv = nn.Conv2d(in_channels=channels,
                                     out_channels=channels,
                                     kernel_size=upsamp_kernel_size,
                                     stride=sampling,
                                     padding=upsamp_kernel_size//2,
                                     bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.down_conv(x)

        return x

class Gaussian_Bicubic_Downsampling(nn.Module):
    def __init__(self, config_data):
        super(Gaussian_Bicubic_Downsampling, self).__init__()
        blur = config_data['blur_sd']
        sampling = config_data['sampling']
        channels = config_data['channels']
        kernel_size = math.ceil(4*blur)+1
        self.gaussian = GaussianSmoothing(channels=channels, kernel_size=kernel_size, sigma=blur)
        self.bicubic = nn.Upsample(scale_factor=1./sampling, mode='bicubic', align_corners=False)
    def forward(self, x):
        x = self.gaussian(x)
        x = self.bicubic(x)

        return x