# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## References
# 
# 1. https://github.com/bentrevett/pytorch-sentiment-analysis

# %%
import numpy as np
import torch

# %% [markdown]
# ## Load data
# 
# TEXT field has tokenize='spacy' as an argument. This defines that the "tokenization" is done using the spaCy tokenizer. If no tokenize argument is passed, the default is simply splitting the string on spaces.
# 
# LABEL is defined by a LabelField, a special subset of the Field class specifically used for handling labels.

# %%
from torchtext import data, datasets

MAX_SEQ_LEN=500

TEXT = data.Field(tokenize='spacy', fix_length=MAX_SEQ_LEN)
LABEL = data.LabelField(dtype=torch.float)
d_train_all, d_test = datasets.IMDB.splits(TEXT, LABEL)


# %%
d_train, d_vali = d_train_all.split(split_ratio=0.8)


# %%
print(f'Number of training examples: {len(d_train)}')
print(f'Number of validation examples: {len(d_vali)}')
print(f'Number of testing examples: {len(d_test)}')

# %% [markdown]
# ## Encoding
# 
# The number of unique words in our training set is over 100,000, which means that our one-hot vectors will have over 100,000 dimensions! This will make training slow and possibly won't fit onto your GPU (if you're using one).
# 
# There are two ways effectively cut down our vocabulary, we can either only take the top $n$ most common words or ignore words that appear fewer than $m$ times. We'll do the former, only keeping the top 25,000 words.

# %%
MAX_VOCAB_SIZE = 5000

TEXT.build_vocab(d_train, max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(d_train)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# %% [markdown]
# ### Most common words

# %%
print(TEXT.vocab.freqs.most_common(20))

# %% [markdown]
# ### Index to string map

# %%
print(TEXT.vocab.itos[:10])

# %% [markdown]
# ### String to index map

# %%
print(LABEL.vocab.stoi)

# %% [markdown]
# ## Training

# %%
BATCH_SIZE = 64

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

d_train_iter, d_vali_iter, d_test_iter = data.BucketIterator.splits(
    (d_train, d_vali, d_test), 
    batch_size = BATCH_SIZE,
    device = device)

# %% [markdown]
# ### RNN

# %%
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        return self.fc(hidden.squeeze(0))

class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden)
        return self.fc(hidden.squeeze(0))

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
OUTPUT_DIM = 1

model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# %% [markdown]
# ### Optimizer

# %%
import torch.optim as optim

# optimizer = optim.SGD(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters())

# %% [markdown]
# ### Criterion

# %%
criterion = nn.BCEWithLogitsLoss()

# %% [markdown]
# Using `.to` to place the model and the criterion on the GPU (if we have one).

# %%
model = model.to(device)
criterion = criterion.to(device)


# %%
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# %%
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        acc = binary_accuracy(predictions, batch.label)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# %%
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# %%
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# %%
N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, d_train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, d_vali_iter, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# %%
for batch in d_train_iter:
    # print(batch)
    # print(batch.text)
    # print(batch.text[:,0])
    # print(batch.text.shape)
    # print(batch.label[0])
    break


# %%



