# %%
import torch
import torch.nn as nn


class TwoLayerNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

    def forward(self, inputs):
        pass


# %%
# number of trainable parameters
def n_params(net):
    nb_params = 0
    for param in net.parameters():
        nb_params += param.numel()
    return nb_params
