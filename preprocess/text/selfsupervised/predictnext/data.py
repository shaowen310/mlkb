# https://github.com/pytorch/examples/tree/master/word_language_model/

import os
from io import open
import torch


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dict = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dict.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dict.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class TextCorpusDatasetCollection:
    def __init__(self, root_dir, window_size):
        '''
        file organization
        Train: root_dir/train.txt
        Test: root_dir/text.txt
        Validation: root_dir/valid.txt
        '''
        self.root_dir = root_dir
        self.corpus = Corpus(root_dir)
        self.train = TextCorpusDataset(self.corpus.train, window_size)
        self.test = TextCorpusDataset(self.corpus.test, window_size)
        self.valid = TextCorpusDataset(self.corpus.valid, window_size)


class TextCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, text_ids, window_size):
        self.data = text_ids
        self.window_size = window_size

    def __getitem__(self, idx):
        return self.data[idx:idx + self.window_size], self.data[idx + self.window_size]

    def __len__(self):
        return len(self.data) - self.window_size
