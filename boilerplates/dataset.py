import torch

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=batch_size)
