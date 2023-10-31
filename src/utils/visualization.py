from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import Tensor


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid



device = "cuda" if torch.cuda.is_available() else "cpu"




class TensorboardWriter:
    def __init__(
            self,
            logdir: str,
            val_loader: DataLoader,
            train_loader: DataLoader,
            device: str,
            name: str,
            std_noise: float | None,
            config: dict,
            model: Any):

        self.config = config
        self.logdir = logdir
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.device = device
        self.model_name = name
        self.std_noise = std_noise
        self.config = config
        self.writer = SummaryWriter(log_dir=logdir)
        self._model_info(model)

    def __call__(
            self,
            train_loss: Tensor,
            val_loss: Tensor,
            train_loss_comp: dict,
            val_loss_comp: dict,
            epoch: int,
            train_metrics: dict,
            train_metrics_uk: dict | None,
            val_metrics: dict,
            val_metrics_uk: dict | None,
            model: Any,
            hyperparameters: Optional[dict] = None,
            add_figures: bool = True,
    ):

        val_metrics = dict(val_metrics)
        train_metrics = dict(train_metrics)

        self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_step=epoch)
        if val_loss_comp:
            self.writer.add_scalars('loss/validation_components', val_loss_comp, global_step=epoch)
        if train_loss_comp:
            self.writer.add_scalars('loss/train_components', train_loss_comp, global_step=epoch)

        self.writer.add_scalar("validation_metrics/PSNR", val_metrics.pop('psnr'),global_step=epoch)
        self.writer.add_scalar("validation_metrics/RMSE", val_metrics.pop('rmse'),global_step=epoch)
        self.writer.add_scalar("validation_metrics/SAM", val_metrics.pop('sam'),global_step=epoch)
        self.writer.add_scalar("validation_metrics/SSIM", val_metrics.pop('ssim'),global_step=epoch)
        self.writer.add_scalar("validation_metrics/ERGAS", val_metrics.pop('ergas'), global_step=epoch)

        self.writer.add_scalar("train_metrics/PSNR", train_metrics.pop('psnr'),global_step=epoch)
        self.writer.add_scalar("train_metrics/RMSE", train_metrics.pop('rmse'),global_step=epoch)
        self.writer.add_scalar("train_metrics/SAM", train_metrics.pop('sam'),global_step=epoch)
        self.writer.add_scalar("train_metrics/SSIM", train_metrics.pop('ssim'),global_step=epoch)
        self.writer.add_scalar("train_metrics/ERGAS", train_metrics.pop('ergas'), global_step=epoch)

        self.writer.add_scalar("parameters/tau", model.tau, global_step=epoch)

        if val_metrics_uk is not None and add_figures:
            x = np.arange(0, len(val_metrics_uk['psnr']))
            # Plot a figure with the evolution of the metrics for each stage
            for k, v in val_metrics_uk.items():
                fig = plt.figure()
                plt.plot(x, v)
                plt.xlabel('stage')
                plt.ylabel(k)
                self.writer.add_figure(f"list_uk_val/{k}", fig, global_step=epoch)
            for k, v in train_metrics_uk.items():
                fig = plt.figure()
                plt.plot(x, v)
                plt.xlabel('stage')
                plt.ylabel(k)
                self.writer.add_figure(f"list_uk_train/{k}", fig, global_step=epoch)




        if hyperparameters is not None:
            self.writer.add_scalars('hyperparameters', hyperparameters, global_step=epoch)

        if add_figures:
            self._add_images(model, epoch)


    def _add_images(self, model, epoch):
        gt_v, low_v, inter_v = next(iter(self.val_loader))
        N = gt_v.shape[0]
        # plot at most 5 images even if the batch size is larger
        if N > 5:
            gt_v = gt_v[:5]
            low_v = low_v[:5]
            inter_v = inter_v[:5]
        with torch.no_grad():
            pred_v, uk = model.forward(low_v.to(self.device).float())
        low_aux = torch.zeros_like(gt_v).to(gt_v.device)
        low_aux[:, :, :low_v.size(2), :low_v.size(3)] = low_v[:, :, :, :]
        self._plot_batch_iter(uk, epoch,'stages/validation')
        self._plot_batch(gt=gt_v, low=low_aux, inter=inter_v, pred=pred_v, epoch=epoch,text='validation_images/Ground_Truth-Low_resolution-Interpolated-Prediction')

        gt_t, down_t, inter_t = next(iter(self.train_loader))
        N = gt_t.shape[0]
        # plot at most 5 images even if the batch size is larger
        if N > 5:
            gt_t = gt_t[:5]
            down_t = down_t[:5]
            inter_t = inter_t[:5]
        with torch.no_grad():
            pred_t, uk = model.forward(down_t.to(self.device).float())
        low_aux = torch.zeros_like(gt_t).to(gt_v.device)
        low_aux[:, :, :down_t.size(2), :down_t.size(3)] = down_t[:, :, :, :]
        self._plot_batch_iter(uk, epoch, 'stages/train')
        self._plot_batch(gt=gt_t, low=low_aux, inter=inter_t, pred=pred_t, epoch=epoch,text='train_images/Ground_Truth-Low_resolution-Interpolated-Prediction')
    def _plot_batch(self,gt, low, inter, pred, epoch, text):
        value_range = (0, 1) if self.config['dataset']['img_range'] == '0_1' else (0, 255)
        gt_v = make_grid(tensor=gt.clamp(0, 1), nrow=1, value_range=value_range).to(self.device)
        low_aux = make_grid(tensor=low.clamp(0, 1), nrow=1, value_range=value_range).to(self.device)
        inter_v = make_grid(tensor=inter.clamp(0, 1), nrow=1, value_range=value_range).to(self.device)
        pred_v = make_grid(tensor=pred.clamp(0, 1), nrow=1, value_range=value_range).to(self.device)
        grid = torch.cat((gt_v.unsqueeze(0), low_aux.unsqueeze(0), inter_v.unsqueeze(0), pred_v.unsqueeze(0)), dim=0)
        self.writer.add_images(text, grid, global_step=epoch)
    def _plot_batch_iter(self,uk, epoch, text):
        value_range = (0, 1) if self.config['dataset']['img_range'] == '0_1' else (0, 255)
        stages = len(uk)
        batch = uk[0].shape[0]
        uk = torch.cat(uk, dim=0)

        grid = make_grid(tensor=uk.clamp(0, 1), nrow=batch, value_range=value_range).to(self.device)
        self.writer.add_image(text, grid, global_step=epoch)

    def _model_info(self, model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.writer.add_text("model info", f"Trainable params: {trainable_params}\nTotal params: {total_params}")
        print(f"Trainable params: {trainable_params}\nTotal params: {total_params}")

    def add_text(self, title, content, step):
        self.writer.add_text(title, content, step)

    def close(self):
        self.writer.close()