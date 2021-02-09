# https://github.com/pytorch/examples/tree/master/word_language_model/
from io import open
import torch
from torch.utils.data import DataLoader


class CorpusDataset(torch.utils.data.Dataset):
    def __init__(self, text_ids, window_size):
        super().__init__()
        self.data = text_ids
        self.window_size = window_size

    def __getitem__(self, idx):
        sid = idx * self.window_size
        eid = sid + self.window_size
        return self.data[sid:eid], self.data[eid]

    def __len__(self):
        return (len(self.data) - 1) // self.window_size


if __name__ == '__main__':
    parsed = False
    corpus = None
    data_dir = ''
    window_size = 1
    batch_size = 128

    if parsed:
        corpus.load(data_dir)
    else:
        corpus.parse(data_dir)
        corpus.save(data_dir)
    ds_train = CorpusDataset(corpus.train, window_size)
    ds_valid = CorpusDataset(corpus.valid, window_size)
    ds_test = CorpusDataset(corpus.test, window_size)

    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=False)
