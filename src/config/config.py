import argparse

import torch
class ParseKwargs(argparse.Action):
    CHOICES = dict()

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.CHOICES)
        for value in values:
            key, value = value.split('=')
            if self.CHOICES and key not in self.CHOICES.keys():
                print(f"{parser.prog}: error: argument {option_string}: invalid choice: '{key}' (choose from {list(self.CHOICES.keys())})")
            else:
                getattr(namespace, self.dest)[key] = self._parse(value)

    @staticmethod
    def _parse(data):
        try:
            return int(data)
        except ValueError:
            pass
        try:
            return float(data)
        except ValueError:
            pass
        return data
class DatasetParsers(ParseKwargs):
    CHOICES = {
        "name": "DIV2K",
        "sampling": 2,
        "blur_sd": 0.7,
        "crop_size": None,
        "img_range": '0_1',
        "channels": 3,
        "std_noise": 25,
    }

class TrainingParsers(ParseKwargs):
    CHOICES =  {
        "epochs": 1000,
        "batch_size": 1,
        "learning_rate": 1.e-3,
        "patience": 10,
        "learning_rate_factor": 0.75,
        "loss_function": "MSE",
        "alpha": 0.1,
        "optimizer": "Adam",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "evaluation_frequency": 5,
    }

class ModelParamsParsers(ParseKwargs):
    CHOICES =  {
        "stages": 4,
        "lmb": 0.01,
        "mu": 0.01,
        "sigma": 0.05,
        "tau": 0.05,
        "resnet_length": 1,
        "resnet_features": 16,
        "resnet_kernel": 3,
        "resner_var_features": 6,
        "resnet_architecture": "Resnet2",
        "windows_size": 5,
        "patch_size": 3,
    }


default_config: dict[str, dict[str, int | float | None | str]] = \
    {
    "dataset": {
        "name": "DIV2K",
        "sampling": 2,
        "blur_sd": 0.7,
        "crop_size": None,
        "img_range": '0_1',
        "channels": 3,
        "std_noise": 25,
        },
    "model": {
        "stages": 4,
        "lmb": 0.01,
        "mu": 0.01,
        "sigma": 0.05,
        "tau": 0.05,
        "resnet_length": 1,
        "resnet_features": 16,
        "resnet_kernel": 3,
        "resner_var_features": 6,
        "resnet_architecture": "Resnet2",
        "windows_size": 5,
        "patch_size": 3,
        },
    "training": {
        "epochs": 1000,
        "batch_size": 1,
        "learning_rate": 1.e-3,
        "patience": 10,
        "learning_rate_factor": 0.75,
        "loss_function": "MSE",
        "alpha": 0.1,
        "optimizer": "Adam",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "evaluation_frequency": 5,
        }
    }