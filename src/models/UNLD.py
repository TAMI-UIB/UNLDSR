import torch
from torch import nn

from src.utils.operators.proxnet import ProxNet

from src.utils.operators.downsampling import LearnedDownsampling
from src.utils.operators.upsampling import LearnedUpsampling

from src.utils.operators.multi_head_attention import MHA





class UNLData(nn.Module):
    def __init__(self,config):
        super(UNLData, self).__init__()
        self.config = config
        self.device = config['training']['device']

        self.sampling = config['dataset']['sampling']
        self.channels = config['dataset']['channels']

        self.stages = config['model']['stages']

        res_args = {'iter': config['model']['resnet_length'], 'channels': self.channels,
                    'features': config['model']['resnet_features'], 'kernel_size': config['model']['resnet_kernel']}
        self.prox = nn.Sequential(*[ProxNet(**res_args) for _ in range(self.stages)])
        self.init = nn.Upsample(scale_factor=config['dataset']['sampling'], mode='bicubic', align_corners=False)
        self.BU = nn.Sequential(*[LearnedUpsampling(self.config['dataset']) for _ in range(self.stages)])

        self.DB = nn.Sequential(*[LearnedDownsampling(self.config['dataset']) for _ in range(self.stages)])
        self.window_size = config['model']['windows_size']
        self.patch_size = config['model']['patch_size']
        self.MHA = nn.Sequential(*[MHA(channels=self.channels, window_size=self.window_size, patch_size=self.patch_size) for _ in range(self.stages)])
        self.tau = nn.Parameter(torch.tensor(config['model']['tau'], device=self.device))
    def forward(self, low):
        u=self.init(low)
        uk=[u]
        for i in range(self.stages):
            dbu = self.DB[i](u)
            low_nl = self.MHA[i](low, dbu)
            argument = u-self.tau*self.BU[i](dbu - low_nl)
            u = self.prox[i](argument)
            uk.append(u)
        return u, uk