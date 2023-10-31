from torch.optim import Adam

from src.utils.loss_functions import MSE

loss_dict = {"MSE": MSE,}

optimizer_dict = {"Adam": Adam}

dict_image_name = {0: 'buda', 1: 'door', 2: 'egipt', 3: 'esquirol', 4: 'paifang', 5: 'penguin'}
