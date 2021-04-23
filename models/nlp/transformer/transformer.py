# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# torch.version >= 1.1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from posencode import PositionalEncoding


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    def __init__(self, vocab_size, emb_dim, n_head, hidden_dim, n_layers, dropout=0.5):
        '''
        '''
        super().__init__()
        self.src_mask = None
        self.emb_dim = emb_dim
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layer = TransformerEncoderLayer(emb_dim, n_head, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
