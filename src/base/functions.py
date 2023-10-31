import os

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils.utils import dict_image_name, save_csv


def training_epoch(model, train_loader, optimizer, loss_f, metrics, metrics_uk,  device, epoch):
    losses = []
    model.train()

    with tqdm(enumerate(train_loader), total=len(train_loader), leave=False) as pbar:
        for idx, batch in pbar:
            optimizer.zero_grad()
            gt, low, inter = batch[0], batch[1], batch[2]
            gt = gt.to(device).float()
            inter = inter.to(device).float()
            low = low.to(device).float()
            pred, list_uk = model.forward(low)

            loss, components = loss_f(pred=pred, gt=gt, list_uk=list_uk)

            loss.backward()
            optimizer.step()
            loss = loss.float()
            losses.append(loss.cpu().detach().numpy())

            pbar.set_description(f"epoch: {epoch} train loss {np.array(losses).mean():.4f}")

            metrics.update(pred.cpu(), gt.cpu(), inter.cpu())
            if metrics_uk is not None:
                for k, uk in enumerate(list_uk):
                    metrics_uk.update(stage=k, uk=uk.cpu(), gt=gt.cpu(), low=low.cpu())

    return np.array(losses).mean(), components, metrics.dict, metrics_uk.dict


def validating_epoch(model, val_loader, loss_f, metrics, metrics_uk, device, epoch):
    with torch.no_grad():
        model.eval()
        losses = []
        with tqdm(enumerate(val_loader), total=len(val_loader), leave=False) as pbar:
            for idx, batch in pbar:
                gt, low, inter = batch[0], batch[1], batch[2]
                gt = gt.to(device).float()
                low = low.to(device).float()
                inter = inter.to(device).float()

                pred, list_uk = model.forward(low)

                loss, components = loss_f(pred=pred, gt=gt, list_uk=list_uk)

                loss = loss.float()
                losses.append(loss.cpu().detach().numpy())
                metrics.update(pred.cpu(), gt.cpu(), inter.cpu())
                if metrics_uk is not None:
                    for k, uk in enumerate(list_uk):
                        metrics_uk.update(stage=k, uk=uk.cpu(), gt=gt.cpu(), low=low.cpu())

                pbar.set_description(f"epoch: {epoch} val loss {np.array(losses).mean():.4f}")

    return np.array(losses).mean(), components, metrics.dict, metrics_uk.dict

def testing(model, data_loader, loss_f, metrics, metrics_uk, device, test_path, config):
    with torch.no_grad():
        model.eval()
        losses = []
        with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as pbar:
            for idx, batch in pbar:
                gt, low, inter = batch[0], batch[1], batch[2]
                gt = gt.to(device).float()
                low = low.to(device).float()
                inter = inter.to(device).float()

                pred, list_uk = model.forward(low)
                csv_path = f"{config['model']['name']}_sampling_{config['dataset']['sampling']}_blur_{config['dataset']['blur_sd']}_noise_{config['dataset']['std_noise']}"
                os.makedirs(test_path + f"/{csv_path}", exist_ok=True)
                loss, components = loss_f(pred=pred, gt=gt, list_uk=list_uk)

                loss = loss.float()
                losses.append(loss.cpu().detach().numpy())
                metric_image = metrics.update(pred.cpu(), gt.cpu(), inter.cpu())
                metric_image.update(dict(image=dict_image_name[idx]))
                if metrics_uk is not None:
                    for k, uk in enumerate(list_uk):
                        metrics_uk.update(stage=k, uk=uk.cpu(), gt=gt.cpu(), low=low.cpu())

                pbar.set_description(f"Loss {np.array(losses).mean():.4f}")

                save_csv(test_path, f"{csv_path}_image_metrics.csv", metric_image)

    return np.array(losses).mean(), components, metrics.dict, metrics_uk.dict


