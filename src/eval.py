import argparse
import os
import sys

from dotenv import load_dotenv

if not os.environ.get('SKIP_DOTENV'):
    load_dotenv()
sys.path.extend([os.environ.get('PROJECT_PATH')])

from src.base.experiment import Experiment
from src.config.config import default_config, TrainingParsers, DatasetParsers, ModelParamsParsers
from src.models import model_dict




def get_config(**args):
    config = dict()
    config.update({'dataset': args['dataset']})
    config.update({'training': args['training']})
    config.update({'unfolding': args['unfolding']})
    config.update({'model': args['model']})
    config.update({'nickname': args['nickname']})
    config.update({'resume_path': args['resume_path']})
    return config

def print_config(config, model):
    print(f"Model: {model}")
    print("Config:")
    for k, v in config.items():
        print(f"\t{k}: {v}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion with unfolding")
    parser.add_argument("--dataset", default=default_config['dataset'], action=DatasetParsers, help="Dataset info")
    parser.add_argument("--training", default=default_config['training'], action=TrainingParsers, help="Training info")
    parser.add_argument("--unfolding", default=default_config['unfolding'], action=ModelParamsParsers, help="Training info")
    parser.add_argument("--model", default="UCLData_PG", type=str, help="Model to train", choices=model_dict.keys())

    parser.add_argument("--nickname", type=str, help="Name for the test")
    parser.add_argument("--resume_path", type=str, help="Resume path")

    args = parser.parse_args()
    config = get_config(**args.__dict__)
    print_config(config, args.model)
    experiment = Experiment(model=args.model, config=config)
    experiment.eval()