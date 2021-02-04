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
    def __init__(self, vocab_size, window_size, embedding_dim, hidden_dim):
        super().__init__()
        self.emb_window_prod = embedding_dim * window_size
        self.C = nn.Embedding(vocab_size, embedding_dim)
        self.H = nn.Linear(self.emb_window_prod, hidden_dim)
        self.U = nn.Linear(hidden_dim, vocab_size, False)
        self.W = nn.Linear(self.emb_window_prod, vocab_size)

    def forward(self, inputs):
        '''
        inputs: word idx seq
        '''
        x = self.C(inputs).view((-1, self.emb_window_prod))
        a = F.tanh(self.H(x))
        y = self.U(a) + self.W(x)
        return y


# %%
nplm = NPLM(5000, 8, 128, 10)

# %%
