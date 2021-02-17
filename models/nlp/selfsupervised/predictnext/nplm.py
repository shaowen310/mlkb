# A Neural Probabilistic Language Model
# Loss: CrossEntropyLoss()
# y = Ua + (Wx + b)
# a = tanh(d + Hx)
# x = (C(w(t-1)), C(w(t-2), ..., C(w(t-n+1))), n == window_size
# C is the embedding matrix
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class NPLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 window_size,
                 embedding_dim,
                 hidden_dim,
                 dropout=0.5,
                 tie_weights=False,
                 direct_connection=False):
        '''
        Args:
            window_size: n in "n-gram"
        '''
        super().__init__()
        self.emb_window_prod = embedding_dim * (window_size - 1)
        self.embl = nn.Embedding(vocab_size, embedding_dim)
        self.H = nn.Linear(self.emb_window_prod, hidden_dim)
        self.U = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            if hidden_dim != embedding_dim:
                raise ValueError(
                    'When using the tied flag, hidden_dim must be equal to embedding_dim')
            self.U.weight = self.embl.weight

        self.direct_connection = direct_connection
        if direct_connection:
            self.W = nn.Linear(self.emb_window_prod, vocab_size, False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        '''
        inputs: word idx seq
        '''
        emb = self.drop(self.embl(inputs)).view((-1, self.emb_window_prod))
        a = self.drop(torch.tanh(self.H(emb)))
        y = self.U(a)
        if self.direct_connection:
            y += self.W(emb)
        return F.log_softmax(y, dim=1)


# %%
nplm = NPLM(5000, 8, 128, 10)

# %%
