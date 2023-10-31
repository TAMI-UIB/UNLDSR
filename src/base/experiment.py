
import os
from datetime import datetime

import torch
from dotenv import load_dotenv

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader




from src.models import model_dict
from src.dataloader import dataset_dict
from src.utils import optimizer_dict, loss_dict

from src.utils.metrics import MetricCalculator, MetricCalculatorStages
from src.utils.utils import save_csv

from src.utils.visualization import TensorboardWriter
from src.base.functions import training_epoch, validating_epoch, testing

if not os.environ.get('SKIP_DOTENV'):
    load_dotenv()

class Experiment:
    def __init__(
            self,
            model: str,
            config: dict
    ) -> None:
        # Constructor to initialize an Experiment object with configuration and model information.
        self.config = config
        self.config.update({'model': {'name': model}})
        self.dataset = dataset_dict[config['dataset']['name']]
        self.model = {'class': model_dict[model], 'name': model}
        self.optimizer = optimizer_dict[config['training']['optimizer']]
        self.device = config['training']['device']
        self.loss = loss_dict[config['training']['loss_function']](config)
        self.eval_n = max(int(config['training']['epochs'] * (self.config['training']['evaluation_frequency'] / 100)),
                          1)
        self.epochs = config['training']['epochs']
        self.snapshot_path = self._get_log_train_path()
        self.test_path = self._get_log_test_path()

        # Set a random seed for reproducibility
        seed = 123
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        # Training function to train the model.
        # It loads data, model, optimizer, and scheduler and then performs training.

        # Load training and validation data
        train_loader, train_len = data_load(dataclass=self.dataset,
                                            dataset_mode='train',
                                            data_info=self.config['dataset'],
                                            batch_size=self.config['training']['batch_size'],
                                            device=self.device,
                                            shuffle=True)
        val_loader, val_len = data_load(dataclass=self.dataset,
                                        dataset_mode='validation',
                                        batch_size=self.config['training']['batch_size'],
                                        data_info=self.config['dataset'],
                                        device=self.device,
                                        shuffle=False)

        # Load model and initialize optimizer and learning rate scheduler
        model, start_epoch = model_load(self.model['class'], resume_path=self.config['resume_path'], config=self.config,
                                        device=self.device)
        optimizer = self.optimizer(model.parameters(), lr=self.config['training']['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, 'min',
                                      factor=self.config['training']['learning_rate_factor'],
                                      patience=self.config['training']['patience'])

        # Create a Tensorboard writer for logging
        writer = TensorboardWriter(logdir=self.snapshot_path,
                                   val_loader=val_loader,
                                   train_loader=train_loader,
                                   device=self.device,
                                   config=self.config,
                                   name=self.model['name'],
                                   model=model,
                                   std_noise=False, )

        # Initialize best PSNR for model saving
        best_psnr = 0.001

        for epoch in range(start_epoch, self.epochs, 1):
            # Initialize metrics for training and validation
            train_metrics = MetricCalculator(train_len, self.config['dataset']['sampling'])
            val_metrics = MetricCalculator(val_len, self.config['dataset']['sampling'])
            if self.model['name'] in our_models:
                train_metrics_uk = MetricCalculatorStages(dataset_len=train_len,
                                                          uk_len=model.stages + 1,
                                                          downsampling=train_loader.dataset.downsampling.forward,
                                                          sampling_factor=self.config['dataset']['sampling'])
                val_metrics_uk = MetricCalculatorStages(dataset_len=val_len,
                                                        uk_len=model.stages + 1,
                                                        downsampling=val_loader.dataset.downsampling.forward,
                                                        sampling_factor=self.config['dataset']['sampling'])
            else:
                train_metrics_uk = None
                val_metrics_uk = None

            # Perform training and validation epochs
            train_loss, train_loss_components, train_metrics, train_metrics_uk = training_epoch(model, train_loader,
                                                                                                optimizer, self.loss,
                                                                                                train_metrics,
                                                                                                train_metrics_uk,
                                                                                                self.device, epoch)
            val_loss, val_loss_components, val_metrics, val_metrics_uk = validating_epoch(model, val_loader, self.loss,
                                                                                          val_metrics, val_metrics_uk,
                                                                                          self.device,
                                                                                          epoch)
            psnr = val_metrics["psnr"]
            scheduler.step(train_loss)

            if epoch % self.eval_n == 0:
                self._print_data(train_loss, val_loss, epoch, val_metrics)
                self._save_model(model, 'last', epoch)
                writer(train_loss=train_loss,
                       val_loss=val_loss,
                       train_loss_comp=train_loss_components,
                       val_loss_comp=val_loss_components,
                       epoch=epoch,
                       train_metrics=train_metrics,
                       train_metrics_uk=train_metrics_uk,
                       val_metrics=val_metrics,
                       val_metrics_uk=val_metrics_uk,
                       model=model)

            if psnr >= best_psnr:
                best_psnr = psnr
                self._save_model(model, 'best', epoch)
                if epoch % self.eval_n != 0:
                    self._print_data(train_loss, val_loss, epoch, val_metrics)
                writer(train_loss=train_loss,
                       val_loss=val_loss,
                       train_loss_comp=train_loss_components,
                       val_loss_comp=val_loss_components,
                       epoch=epoch,
                       train_metrics=train_metrics,
                       train_metrics_uk=train_metrics_uk,
                       val_metrics=val_metrics,
                       val_metrics_uk=val_metrics_uk,
                       model=model,
                       add_figures=False)
                writer.add_text("best metrics epoch in validation", str(val_metrics), epoch)
        self._save_model(model, 'last', self.epochs)
        writer.close()

    def eval(self):
        # Evaluation function to evaluate the trained model.
        # It loads data, model, and performs evaluation.

        data_loader, data_len = data_load(dataclass=self.dataset,
                                          dataset_mode='validation',
                                          batch_size=self.config['training']['batch_size'],
                                          data_info=self.config['dataset'],
                                          device=self.device,
                                          shuffle=False)
        model, weights_epoch = model_load(self.model['class'], resume_path=self.config['resume_path'],
                                          config=self.config, device=self.device)
        if self.model['name'] in our_models:
            metrics_uk = MetricCalculatorStages(dataset_len=data_len,
                                                uk_len=model.stages + 1,
                                                downsampling=data_loader.dataset.downsampling.forward,
                                                sampling_factor=self.config['dataset']['sampling'])
        else:
            metrics_uk = None
        metrics = MetricCalculator(data_len, self.config['dataset']['sampling'])
        loss, loss_components, metrics, metrics_uk = testing(model=model,
                                                             data_loader=data_loader,
                                                             loss_f=self.loss,
                                                             metrics=metrics,
                                                             metrics_uk=metrics_uk,
                                                             device=self.device,
                                                             test_path=self.test_path,
                                                             config=self.config)

        csv_name = f"sampling_{self.config['dataset']['sampling']}_blur_{self.config['dataset']['blur_sd']}_mean_metrics.csv"

        save_dict = dict(model=self.model['name'], **metrics, epoch=weights_epoch, loss=loss,
                         noise=self.config['dataset']['std_noise'], stages=self.config['unfolding']['stages'])
        save_csv(self.test_path, csv_name, save_dict)

    def _save_model(self, model, version, epoch=None):
        # Private method to save the model's weights at a given version and epoch.
        try:
            os.makedirs(self.snapshot_path + '/ckpt/')
        except FileExistsError:
            pass
        save_path = self.snapshot_path + f'/ckpt/weights_{version}.pth'
        ckpt = {'weights': model.state_dict(), 'epoch': epoch}
        torch.save(ckpt, save_path)

    @staticmethod
    def _print_data(train_loss, val_loss, epoch, metrics):
        # Private method to print training and validation data.
        metrics_message = ", ".join([f"{k} {v:.2f}" for k, v in metrics.items()])
        print(f"epoch {epoch}: trainloss {train_loss:.4f}, valloss {val_loss:.4f}, {metrics_message}")

    def _get_log_train_path(self):
        # Private method to get the path for logging training information.
        now = datetime.now()
        date = now.strftime("%Y-%m-d")
        loss_function = self.config['training']['loss_function']
        nickname = self.config['nickname']
        name = f"model-{self.model['name']}_loss-{loss_function}_arch-{self.config['unfolding']['resnet_architecture']}_stages-{self.config['unfolding']['stages']}"
        snapshot_path = os.environ["SNAPSHOT_PATH"] + f"/{self.config['dataset']['name']}/{date}/{name}" if nickname is None else os.environ["SNAPSHOT_PATH"] + f"/{self.config['dataset']['name']}/{nickname}/{name}"
        return snapshot_path

    def _get_log_test_path(self):
        # Private method to get the path for logging test information.
        path = os.environ["SNAPSHOT_PATH"] + f"/{self.config['dataset']['name']}/{self.config['nickname']}"
        return path

    def data_load(self, dataclass, dataset_mode, data_info, batch_size, device, shuffle=True):
        pin_memory = True if device == 'cuda' else False
        dataset_path = os.environ["DATASET_PATH"] + f"/{data_info['name']}/"
        dataset = dataclass(dataset_path=dataset_path,
                            dataset_mode=dataset_mode,
                            dataset_info=data_info, )
        data_len = len(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=int(os.environ.get("NUM_WORKERS", 2)),
                                pin_memory=pin_memory)

        return dataloader, data_len

    def model_load(self, modelclass, device: str, config: dict, resume_path: str | None):
        model = modelclass(config)
        if resume_path is not None:
            ckpt = torch.load(resume_path, map_location=device)
            epoch0 = ckpt['epoch']
            model.load_state_dict(ckpt['weights'])
        else:
            epoch0 = 1
        model = model.float()
        model.to(device)
        return model, epoch0
