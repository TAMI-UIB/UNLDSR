import torch
from torch import nn


from src.utils.operators.downsampling import  LearnedDownsampling
from src.utils.operators.upsampling import LearnedUpsampling

from src.utils.operators.proxnet import ProxNet


class UCLData(nn.Module):
    def __init__(self,config):
        super(UCLData, self).__init__()
        self.config = config
        self.device = config['training']['device']

        self.sampling = config['dataset']['sampling']
        self.channels = config['dataset']['channels']

        self.stages = config['unfolding']['stages']

        res_args = {'iter': config['unfolding']['resnet_length'], 'channels': self.channels,
                    'features': config['unfolding']['resnet_features'], 'kernel_size': config['unfolding']['resnet_kernel']}
        self.prox = nn.Sequential(*[ProxNet(**res_args) for _ in range(self.stages)])
        self.init = nn.Upsample(scale_factor=config['dataset']['sampling'], mode='bicubic', align_corners=False)
        self.BU = nn.Sequential(*[LearnedUpsampling(self.config['dataset']) for _ in range(self.stages)])

        self.DB = nn.Sequential(*[LearnedDownsampling(self.config['dataset']) for _ in range(self.stages)])


        self.tau = nn.Parameter(torch.tensor(config['unfolding']['tau'], device=self.device))
    def forward(self, low):
        u=self.init(low)
        uk=[u]
        for i in range(self.stages):
            argument = u-self.tau*self.BU[i](self.DB[i](u) - low)
            u = self.prox[i](argument)
            uk.append(u)
        return u, uk

