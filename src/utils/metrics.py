import numpy as np
import torch


from torchmetrics.functional.image import structural_similarity_index_measure as SSIM


def ERGAS(pred, target, sampling_factor):
    channel_rmse = torch.mean(torch.sqrt(torch.mean(torch.square(pred - target), dim=(2, 3))))
    channel_mean = torch.mean(pred, dim=(2, 3))
    channel_sum = torch.mean(torch.div(channel_rmse, channel_mean) ** 2, dim=1)
    return 100 * sampling_factor * torch.mean(torch.sqrt(channel_sum))


def RMSE(pred, target):
    return torch.mean(torch.sqrt(torch.mean(torch.square(pred - target), dim=(1, 2, 3))))

def MSE(pred, target):
    return torch.mean(torch.square(pred - target))
def PSNR(pred, target):
    psnr_list = -10 * torch.log10(torch.mean(torch.square(pred - target), dim=(1, 2, 3)))
    return torch.mean(psnr_list)


def SAM(pred, target):
    scalar_dot = torch.sum(torch.mul(pred, target), dim=(1, 2, 3), keepdim=True)
    norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=(1, 2, 3), keepdim=True))
    norm_target = torch.sqrt(torch.sum(target ** 2, dim=(1, 2, 3), keepdim=True))
    return torch.mean(torch.arccos(scalar_dot / (norm_pred * norm_target)))


class MetricCalculator:
    def __init__(self, dataset_len, sampling_factor=4):

        self.len = dataset_len
        self.dict = {'ergas': 0, 'rmse': 0, 'psnr': 0, 'psnr_inter_pred': 0, 'ssim': 0, 'sam': 0, 'psnr_inter_gt': 0}
        self.sampling_factor = sampling_factor


    def update(self, pred, target, inter=None) -> dict:

        rmse = RMSE(pred, target).item()
        psnr = PSNR(pred, target).item()
        ergas = ERGAS(pred, target, self.sampling_factor).item()
        ssim = SSIM(pred, target, data_range=1.).item()
        sam = SAM(pred, target).item()

        N = pred.shape[0]

        self.dict['sam'] += N * sam / self.len
        self.dict['ergas'] += N * ergas / self.len
        self.dict['rmse'] += N * rmse / self.len
        self.dict['psnr'] += N * psnr / self.len
        self.dict['ssim'] += N * ssim / self.len

        if inter is not None:
            psnr_inter_pred = PSNR(pred, inter).item()
            psnr_inter_gt = PSNR(target, inter).item()
            self.dict['psnr_inter_pred'] += N * psnr_inter_pred / self.len
            self.dict['psnr_inter_gt'] += N * psnr_inter_gt / self.len

        return {'ergas': ergas, 'psnr': psnr, 'rmse': rmse, 'ssim': ssim, 'sam': sam}

class MetricCalculatorStages:
    def __init__(self, dataset_len,uk_len, downsampling, sampling_factor=4):

        self.len = dataset_len
        self.dict = {'ergas': np.zeros(uk_len),
                     'rmse': np.zeros(uk_len),
                     'psnr': np.zeros(uk_len),
                     'ssim': np.zeros(uk_len),
                     'sam': np.zeros(uk_len),
                     'energy': np.zeros(uk_len)}
        self.sampling_factor = sampling_factor
        self.DB = downsampling



    def update(self, stage, uk, gt, low ):

        rmse = RMSE(uk, gt).item()
        psnr = PSNR(uk, gt).item()
        ergas = ERGAS(uk, gt, self.sampling_factor).item()
        ssim = SSIM(uk, gt, data_range=1.).item()
        sam = SAM(uk, gt).item()

        N = uk.shape[0]

        self.dict['sam'][stage] += N * sam / self.len
        self.dict['ergas'][stage] += N * ergas / self.len
        self.dict['rmse'][stage] += N * rmse / self.len
        self.dict['psnr'][stage]+= N * psnr / self.len
        self.dict['ssim'][stage] += N * ssim / self.len
        self.dict['energy'][stage] += N * self.compute_energy(u=uk, f=low) / self.len

    def compute_energy(self, u, f):
        return MSE(self.DB(u), f)