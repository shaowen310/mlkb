# A Neural Probabilistic Language Model
# Loss: CrossEntropyLoss()
# y = Ua + (Wx + b)
# a = tanh(d + Hx)
# x = (C(w(t-1)), C(w(t-2), ..., C(w(t-n+1))), n == window_size
# C is the embedding matrix
# %%
import torch.nn as nn
import torch.nn.functional as F


# %%
class NPLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.C = nn.Embedding(vocab_size, embedding_dim)
        self.H = nn.Linear(embedding_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, out_dim, False)
        self.W = nn.Linear(embedding_dim, out_dim)

    def forward(self, inputs):
        '''
        inputs: word idx seq
        '''
        x = self.C(inputs)
        a = F.tanh(self.H(x))
        y = self.U(a) + self.W(x)
        return y


# %%
nplm = NPLM(5000, 256, 600, 10)
