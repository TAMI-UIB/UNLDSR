import csv
import os

import torch
from PIL import Image


dict_image_name = {0: 'buda', 1: 'door', 2: 'egipt', 3: 'esquirol', 4: 'paifang', 5: 'penguin'}
def save_csv(path, name, data):
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name)
    exists = os.path.isfile(file)
    with open(file, 'a', newline='') as f:
        row = dict(**data)
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def save_png_img_0_255(tensor, path, tensor_format):
    match tensor_format:
        case 'NCHW':
            if tensor.size(0)==1:
                tensor = tensor[0].clamp(0,255).type(torch.uint8).cpu()
                ndarr = tensor.permute(1,2,0).numpy()
                im = Image.fromarray(ndarr)
                im.save(path)
            else:
                raise ValueError('Batch dimension must be 1')
        case other:
            raise ValueError('Only NCHW format supported')