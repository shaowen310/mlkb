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
    def __init__(self, vocab_size, window_size, embedding_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.emb_window_prod = embedding_dim * window_size
        self.embl = nn.Embedding(vocab_size, embedding_dim)
        self.H = nn.Linear(self.emb_window_prod, hidden_dim)
        self.U = nn.Linear(hidden_dim, vocab_size, False)
        self.W = nn.Linear(self.emb_window_prod, vocab_size)
        self.drop = nn.Dropout(dropout)

        self.__init_weights()

    def __init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embl.weight, -initrange, initrange)

    def forward(self, inputs):
        '''
        inputs: word idx seq
        '''
        emb = self.drop(self.embl(inputs)).view((-1, self.emb_window_prod))
        a = self.drop(torch.tanh(self.H(emb)))
        y = self.U(a) + self.W(emb)
        return F.log_softmax(y, dim=1)


# %%
nplm = NPLM(5000, 8, 128, 10)

# %%
