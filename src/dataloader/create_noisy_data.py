import argparse

import os
import sys
from dotenv import load_dotenv
if not os.environ.get('SKIP_DOTENV'):
    load_dotenv()
sys.path.extend([os.environ.get('PROJECT_PATH')])
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn

from src.utils.operators.downsampling import Gaussian_Bicubic_Downsampling
from src.utils.utils import save_png_img_0_255



seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def add_noise(tensor, std_noise, device='cpu'):
    return tensor + torch.normal(0, std_noise, size=tensor.shape).to(device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unfolding parameters')
    parser.add_argument('--dataset', type=str, default='DIV2K', help='Dataset name')
    parser.add_argument('--sampling', type=int, default=2, help='Sampling factor')
    parser.add_argument('--blur_sd', type=float, default=0.7, help='Standard deviation of the Gaussian blur')
    parser.add_argument('--std_noise', type=float, default=0., help='Standar desviation of the Gaussian noise')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels')
    parser.add_argument('--dataset_mode', type=str, default='validation', help='Datsetmode')

    arg = parser.parse_args()
    Downsampling = Gaussian_Bicubic_Downsampling(arg.__dict__)
    Bicubic = nn.Upsample(scale_factor=arg.sampling, mode='bicubic', align_corners=False)
    dataset_path = os.environ["DATASET_PATH"] + f"/{arg.dataset}/"
    images_folder = dataset_path + arg.dataset_mode + "/"
    images = sorted([img for img in os.listdir(images_folder) if img.endswith(".png")])
    os.makedirs(os.environ["DATASET_PATH"] + f"/{arg.dataset}/{arg.dataset_mode}/sampling_{arg.sampling}_blur_{arg.blur_sd}_noise_{arg.std_noise}/low/", exist_ok=True)
    os.makedirs(os.environ["DATASET_PATH"] + f"/{arg.dataset}/{arg.dataset_mode}/sampling_{arg.sampling}_blur_{arg.blur_sd}_noise_{arg.std_noise}/inter/",exist_ok=True)
    for img in images:
        path = os.environ["DATASET_PATH"] + f"/{arg.dataset}/{arg.dataset_mode}/" + img
        img_pil = Image.open(path)
        gt = TF.pil_to_tensor(img_pil).to(torch.float)
        low = add_noise(Downsampling(gt.unsqueeze(0)), arg.std_noise)
        name = img.split(".")[0]
        save_path = os.environ["DATASET_PATH"] + f"/{arg.dataset}/{arg.dataset_mode}/sampling_{arg.sampling}_blur_{arg.blur_sd}_noise_{arg.std_noise}/low/" + name + ".png"
        save_png_img_0_255(low, save_path, 'NCHW')
        inter = Bicubic(low)
        save_path = os.environ["DATASET_PATH"] + f"/{arg.dataset}/{arg.dataset_mode}/sampling_{arg.sampling}_blur_{arg.blur_sd}_noise_{arg.std_noise}/inter/" + name + ".png"
        save_png_img_0_255(inter, save_path, 'NCHW')