# %%
import torch

# %%
# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(torch.cuda.device_count() - 1))
else:
    device = torch.device('cpu')

torch.cuda.set_device(torch.cuda.device_count() - 1)
