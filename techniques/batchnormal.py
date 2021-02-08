# %%
import torch.nn as nn
import torch.nn.functional as F


# %%
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Default PyTorch layer is initialized with a normal distribution
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        # nn.init.xavier_normal_(self.hidden_layer.weight)  # Xavier Initialization: UNIFORM between +- sqrt(6 / (n_input + n_output))
        nn.init.kaiming_normal_(
            self.linear0.weight)  # He Initialization: NORMAL with std sqrt(2/n_input)
        self.linear1 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        # self.batchnorm = nn.BatchNorm1d(num_features=hidden_dim)

    def forward(self, inputs):
        out = self.linear0(inputs)
        # out = self.batchnorm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear1(out)
        return out
