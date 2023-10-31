import math
from torch import nn
from sympy import factorint
from src.utils.operators.gaussian_blur import LearnedGaussianSmoothing
class LearnedUpsampling(nn.Module):
    def __init__(self, config_data):
        super(LearnedUpsampling, self).__init__()
        sampling, blur, channels = config_data['sampling'], config_data['blur_sd'], config_data['channels']

        kernel_size = math.ceil(4*blur)+1
        up_layers = []
        for p, exp in factorint(sampling).items():
            for _ in range(exp):
                kernel = p+1 if p % 2 == 0 else p+2
                up_layers.append(nn.ConvTranspose2d(in_channels=channels,
                                     out_channels=channels,
                                     kernel_size=kernel,
                                     stride=p,
                                     padding=kernel//2,
                                     bias=False,
                                    output_padding=p-1))
        self.up_conv = nn.Sequential(*up_layers)
        self.conv = LearnedGaussianSmoothing(channels=channels, kernel_size=kernel_size, sigma=blur)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.conv(x)
        return x