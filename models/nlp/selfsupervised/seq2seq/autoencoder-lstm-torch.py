# %%
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader

# %%
device = 'cuda'  #'cpu'


# %%
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional, dropout=0.2):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            n_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, bidirectional, dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size,
                            output_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional=False):
        super(AutoEncoderLSTM, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, n_layers, bidirectional)
        self.decoder = DecoderLSTM(hidden_size, input_size, n_layers, bidirectional)

    def forward(self, x):
        encoded_x = self.encoder(x)
        encoded_expanded_x = encoded_x.expand(-1, x.shape[1], -1)
        decoded_x = self.decoder(encoded_expanded_x)

        return decoded_x


# %%
def train_eval(model, dl, optimizer, criterion, phase):
    epoch_loss = 0.
    epoch_n_x = 0

    if phase == 'train':
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(phase == 'train'):
        for x, in dl:
            seq_len = x.shape[1]
            epoch_n_x += x.shape[0]

            x = x.to(device)

            if phase == 'train':
                optimizer.zero_grad()

            x_out = model(x)
            inv_idx = torch.arange(seq_len - 1, -1, -1).long()

            diff = (x_out - x[:, inv_idx, :]).detach().cpu().numpy()
            print(diff)
            print(np.sqrt(np.sum(diff**2)))
            print(np.linalg.norm(diff))

            loss = criterion(x_out, x[:, inv_idx, :])

            if phase == 'train':
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * x.shape[0]

    return epoch_loss / epoch_n_x


# %%
sequence = np.arange(50) / 50
n_features = 5
seq_len = len(sequence) // n_features
batch_size = 50

sequence = sequence.reshape((1, seq_len, n_features))

x_train = torch.as_tensor(sequence, dtype=torch.float)
d_train = TensorDataset(x_train)
dl_train = DataLoader(d_train, batch_size=batch_size)

# %%
n_hidden_units = 100
n_layers = 2
learning_rate = 1e-3

model = AutoEncoderLSTM(n_features, n_hidden_units, n_layers)
criterion = nn.MSELoss()
model = model.to(device)
criterion = criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
n_epochs = 50


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


for epoch in range(n_epochs):

    start_time = time.time()

    train_loss = train_eval(model, dl_train, optimizer, criterion, 'train')

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')

# %%
