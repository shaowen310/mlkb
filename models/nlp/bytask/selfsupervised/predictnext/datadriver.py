# %%
import data
import torch

# %%
ds_collection = data.TextCorpusDatasetCollection('./data/wikitext-2', 5)
ds_train = ds_collection.train

# %%
train_loader = torch.utils.data.DataLoader(dataset=ds_train, batch_size=32, shuffle=False)

for seqs, target in train_loader:
    break
# %%
