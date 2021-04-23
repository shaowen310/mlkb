import os
import pickle

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

    def save(self, fp):
        pickle.dump(self.idx2word, open(fp, 'wb'))

    def load(self, fp):
        self.idx2word = pickle.load(open(fp, 'rb'))
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}


class Corpus:
    def __init__(self):
        self.dict = Dictionary()
        self.train = torch.tensor(())
        self.valid = torch.tensor(())
        self.test = torch.tensor(())

    def parse(self, path):
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
                words = line.lower().split() + ['<eos>']
                for word in words:
                    self.dict.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.lower().split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dict.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    def save(self, root_dir):
        self.dict.save(os.path.join(root_dir, 'dict.pickle'))
        torch.save(self.train, os.path.join(root_dir, 'train.pt'))
        torch.save(self.valid, os.path.join(root_dir, 'valid.pt'))
        torch.save(self.test, os.path.join(root_dir, 'test.pt'))

    def load(self, root_dir):
        self.dict.load(os.path.join(root_dir, 'dict.pickle'))
        self.train = torch.load(os.path.join(root_dir, 'train.pt'))
        self.valid = torch.load(os.path.join(root_dir, 'valid.pt'))
        self.test = torch.load(os.path.join(root_dir, 'test.pt'))
