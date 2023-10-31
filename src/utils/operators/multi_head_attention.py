import torch
from torch import nn
import torch.nn.functional as F

class SpatialWeightsEsDot(torch.nn.Module):
    def __init__(self, channels, window_size, patch_size):
        super(SpatialWeightsEsDot, self).__init__()
        self.channels = channels
        self.phi = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Conv2d(channels, channels, 1, bias=False)
        self.window_size = window_size
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-6

    def forward(self, u):
        b, c, h, w = u.size()
        phi = self.phi(u)
        # self.phi(u)
        theta = self.phi(u)
        # self.theta(u)
        theta = F.unfold(theta, self.patch_size, padding=self.patch_size // 2)
        theta = theta.view(b, 1, c * self.patch_size * self.patch_size, -1)
        theta = theta.view(b, 1, c * self.patch_size * self.patch_size, h, w)
        theta = theta.permute(0, 3, 4, 1, 2)

        phi = F.unfold(phi, self.patch_size, padding=self.patch_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, h, w)
        phi = F.unfold(phi, self.window_size, padding=self.window_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, self.window_size * self.window_size, h, w)
        phi = phi.permute(0, 3, 4, 1, 2)

        att = torch.matmul(theta, phi)

        return self.softmax(att)

class SelfAttentionEsDot(torch.nn.Module):
    def __init__(self, channels, patch_size, window_size, aux_channels=None):
        super(SelfAttentionEsDot, self).__init__()
        if aux_channels is None:
          aux_channels=channels
        self.channels = channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.spatial_weights = SpatialWeightsEsDot(channels=aux_channels, window_size=window_size, patch_size=patch_size)
        self.g = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, u, f):
        b, c, h, w = u.size()
        weights = self.spatial_weights(f)
        g = self.g(u)  # [b, 3, h, w]
        g = F.unfold(g, self.window_size, padding=self.window_size // 2)
        g = g.view(b, self.channels, self.window_size * self.window_size, -1)
        g = g.view(b, self.channels, self.window_size * self.window_size, h, w)
        g = g.permute(0, 3, 4, 2, 1)
        return torch.matmul(weights, g).permute(0, 4, 1, 2, 3)

class MHA(torch.nn.Module):
    def __init__(self, channels, patch_size, window_size):
        super(MHA, self).__init__()
        self.geometric_head = SelfAttentionEsDot(channels=channels, patch_size=patch_size,
                                                 window_size=window_size)
        self.spectral_head = SelfAttentionEsDot(channels=channels, patch_size=1,
                                                window_size=window_size)
        self.mix_head = SelfAttentionEsDot(channels=channels, aux_channels=channels+channels, patch_size=patch_size, window_size=window_size)
        self.mlp = nn.Linear(3, 1)

    def forward(self, u, f):
        head1 = self.geometric_head(u, f)
        head2 = self.spectral_head(u, u)
        head3 = self.mix_head(u, torch.concat([u, f], dim=1))
        return self.mlp(torch.concat([head1, head2, head3], dim=4)).squeeze(4)