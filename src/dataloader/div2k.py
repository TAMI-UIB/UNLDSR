import os
from typing import Literal, Tuple, List
import torch
from torch import Tensor, nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from src.utils.operators.downsampling import Gaussian_Bicubic_Downsampling


class DIV2K(Dataset):

    def __init__(
            self,
            dataset_path: str | os.PathLike,
            dataset_mode: Literal['train', 'validation', 'test', 'buda'],
            dataset_info: dict[str, int | str | float],
            bicubic_downsampling: bool = False
    ) -> None:
        super().__init__()

        self.crop_size = dataset_info['crop_size']
        self.sampling = dataset_info['sampling']
        self.img_range = dataset_info['img_range']
        self.blur_sd = dataset_info['blur_sd']
        self.channels = dataset_info['channels']
        self.std_noise = dataset_info['std_noise']
        self.downsampling = Gaussian_Bicubic_Downsampling(dataset_info)
        self.bicubic = nn.Upsample(scale_factor=self.sampling, mode='bicubic', align_corners=False)

        self.mode = dataset_mode
        self.images_folder = dataset_path + dataset_mode
        self.images = sorted([img for img in os.listdir(self.images_folder) if img.endswith('.png')])

        self.gt, self.low, self.inter = self.generate_samples()


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.gt[index], self.low[index], self.inter[index]

    def __len__(self) -> int:
        return len(self.gt)
    def lenght(self) -> int:
        return len(self.gt)
    def generate_samples(self):
        all_crops = []
        low_crops = []
        inter_crops = []
        if self.mode == 'train':
            for path in self.images:
                path = self.images_folder +"/" +path
                image = self.read_image(path)
                if image.size(1) % self.sampling != 0 or image.size(2) % self.sampling != 0:
                    image = image[:, :image.size(1) - image.size(1) % self.sampling,
                            :image.size(2) - image.size(2) % self.sampling]
                image_crops = self.get_crops(image)
                all_crops.extend(image_crops)
            for id, crop in enumerate(all_crops):
                low = self.downsampling(crop.unsqueeze(0))
                low = low + torch.randn(low.size()) * self.std_noise
                inter = self.bicubic(low).squeeze(0)
                low_crops.append(low.squeeze(0))
                inter_crops.append(inter.squeeze(0))

        if self.mode == 'validation':
            for path in self.images:
                path_gt = self.images_folder + "/" +path
                image = self.read_image(path_gt)
                if image.size(1) % self.sampling != 0 or image.size(2) % self.sampling != 0:
                    image = image[:, :image.size(1) - image.size(1) % self.sampling,
                            :image.size(2) - image.size(2) % self.sampling]
                image_crops = self.get_crops(image)
                all_crops.extend(image_crops)
                if self.sampling == 2 and self.blur_sd == 0.7:
                    path_low = f'{self.images_folder}/noise_{self.std_noise}/low/{path}'
                    low_image = self.read_image(path_low)
                    low_crops.extend(self.get_crops(low_image, low=True))
                    path_inter = f'{self.images_folder}/noise_{self.std_noise}/inter/{path}'
                else:
                    path_low = f'{self.images_folder}/sampling_{self.sampling}_blur_{self.blur_sd}_noise_{self.std_noise}/low/{path}'
                    low_image = self.read_image(path_low)
                    low_crops.extend(self.get_crops(low_image, low=True))
                    path_inter = f'{self.images_folder}/sampling_{self.sampling}_blur_{self.blur_sd}_noise_{self.std_noise}/inter/{path}'
                inter_image = self.read_image(path_inter)
                inter_crops.extend(self.get_crops(inter_image))
        if self.img_range == '0_1':
            for id,low in enumerate(low_crops):
                low_crops[id] = low/255.
            for id, inter in enumerate(inter_crops):
                inter_crops[id] = inter/255.
            for id, gt in enumerate(all_crops):
                all_crops[id] = gt/255.
        return all_crops, low_crops, inter_crops

    def get_crops(self, img: Tensor, low=False) -> List[Tensor]:
        if self.crop_size is not None:
            if not low:
                crops = [img[:, i:i + self.crop_size, j:j + self.crop_size] for i in range(0, img.shape[-2], self.crop_size)
                         for j in range(0, img.shape[-1], self.crop_size)]
            else:
                crops = [img[:, i:i + self.crop_size // self.sampling, j:j + self.crop_size // self.sampling] for i in range(0, img.shape[-2], self.crop_size // self.sampling)
                         for j in range(0, img.shape[-1], self.crop_size // self.sampling)]
        else:
            crops = [img]
        return crops

    def read_image(self, img_path: str) -> Tensor:
        img_pil = Image.open(img_path)
        return TF.pil_to_tensor(img_pil).to(torch.float)