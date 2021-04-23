# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
    Args:
        rnn_type: any value of ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']
    """
    def __init__(self,
                 rnn_type,
                 vocab_size,
                 emb_dim,
                 hidden_dim,
                 n_layers,
                 dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emb_dim,
                              hidden_dim,
                              n_layers,
                              nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_dim != emb_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, bsz, self.hidden_dim),
                    weight.new_zeros(self.n_layers, bsz, self.hidden_dim))
        else:
            return weight.new_zeros(self.n_layers, bsz, self.hidden_dim)


if __name__ == '__main__':
    lstm = RNNModel('LSTM')
    batch_size = 128

    def train(model, dataloader):
        hidden = model.init_hidden(dataloader.dataset.batch_size)

        for batch, (elems, labels) in enumerate(dataloader):
            preds, hidden = model(elems, hidden)

            hidden = repackage_hidden(hidden)
