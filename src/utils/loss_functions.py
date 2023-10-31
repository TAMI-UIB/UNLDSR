import torch.nn as nn


class MSE(nn.Module):
    def __init__(self, config):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, gt, list_uk):
        mse_gt = self.mse(pred, gt)
        return mse_gt, None