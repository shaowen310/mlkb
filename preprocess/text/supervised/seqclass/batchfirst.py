import torch


class CharCNNDataset(torch.utils.data.Dataset):
    def __init__(self, char_seqs, labels):
        super().__init__()
        self.char_seqs = char_seqs
        self.labels = labels

    def __getitem__(self, idx):
        return self.char_seqs[idx], self.labels[idx]

    def __len__(self):
        return len(self.char_seqs)
