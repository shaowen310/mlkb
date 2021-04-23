import math
import time
import os

import torch

from corpus import Corpus
from batchify import CorpusBatchifyBPTTDataset
from transformer import TransformerModel

# %%
seed = 11
torch.manual_seed(seed)

bs = 64
max_seq_len = 35

# %%
# Load data
datadir = os.path.join('data_', 'wikitext-2')

data = Corpus()
data.load(datadir)

d_train = data.train
d_valid = data.valid
ds_train = CorpusBatchifyBPTTDataset(d_train, bs, max_seq_len)
ds_valid = CorpusBatchifyBPTTDataset(d_valid, bs, max_seq_len)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=None, shuffle=True)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=None)
# %%
# Model
device = torch.device('cuda:1')

vocab_size = len(data.dict)
word_emb_size = 200
n_heads = 2
hidden_size = 200
n_layers = 2
drop = 0.2

lr = 0.0001

model = TransformerModel(vocab_size, word_emb_size, n_heads, hidden_size, n_layers, drop)

criterion = torch.nn.CrossEntropyLoss()

model.to(device)
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# %%
def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device, log_interval=1000):
    model.to(device)
    criterion.to(device)

    model.train()

    epoch_loss = 0
    log_loss = 0
    start_time = time.time()

    for batch, (elems, labels) in enumerate(dataloader):
        elems = elems.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(elems)
        preds = preds.view(-1, vocab_size)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        log_loss += batch_loss

        if log_interval and batch % log_interval == 0 and batch > 0:
            cur_loss = log_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | {:5.2f} ms/batch  | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(epoch + 1, batch + 1,
                                                        elapsed * 1000 / log_interval, cur_loss,
                                                        math.exp(cur_loss)))
            log_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader) / bs


# %%
for epoch in range(5):
    epoch_loss = train_one_epoch(epoch, model, dl_train, optimizer, criterion, device)
    print('| epoch {} | epoch_loss {} |'.format(epoch + 1, epoch_loss))
# %%
