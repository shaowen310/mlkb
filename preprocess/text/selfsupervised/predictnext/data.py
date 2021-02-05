# https://github.com/pytorch/examples/tree/master/word_language_model/

import os
from io import open
import torch
import pickle


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
        pickle.save(self.idx2word, open(fp, 'wb'))

    def load(self, fp):
        self.idx2word = pickle.load(open(fp, 'rb'))
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}


class Corpus:
    def __init__(self):
        self.dict = Dictionary()
        self.train = torch.tensor(())
        self.valid = torch.tensor(())
        self.text = torch.tensor(())

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

    def save(self, root_dir):
        self.dict.save(os.path.join(root_dir, 'dict.pickle'))
        torch.save(self.train, os.path.join(root_dir, 'train.pt'))
        torch.save(self.valid, os.path.join(root_dir, 'valid.pt'))
        torch.save(self.test, os.path.join(root_dir, 'test.pt'))

    def load(self, root_dir):
        self.dict.load(os.path.join(root_dir, 'dict.pickle'))
        self.train = torch.load(os.path.join(root_dir, 'trian.pt'))
        self.valid = torch.load(os.path.join(root_dir, 'valid.pt'))
        self.test = torch.load(os.path.join(root_dir, 'test.pt'))


class TextCorpusDatasetCollection:
    def __init__(self):
        self.root_dir = ''
        self.corpus = None
        self.train = None
        self.test = None
        self.valid = None

    def init_datasets(self):
        self.train = TextCorpusDataset(self.corpus.train, self.window_size)
        self.test = TextCorpusDataset(self.corpus.test, self.window_size)
        self.valid = TextCorpusDataset(self.corpus.valid, self.window_size)

    def parse(self, root_dir, window_size):
        '''
        file organization
        Train: root_dir/train.txt
        Test: root_dir/test.txt
        Validation: root_dir/valid.txt
        '''
        self.root_dir = root_dir
        self.window_size = window_size
        self.corpus = Corpus()
        self.corpus.parse(root_dir)
        self.init_datasets()

    def load(self, root_dir, window_size):
        '''
        file organization
        Dictionary: root_dir/dict.pickle
        Train: root_dir/train.pt
        Test: root_dir/test.pt
        Validation: root_dir/valid.pt
        '''
        self.root_dir = root_dir
        self.window_size = window_size
        self.corpus.load(root_dir)
        self.init_datasets()


class TextCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, text_ids, window_size):
        self.data = text_ids
        self.window_size = window_size

    def __getitem__(self, idx):
        return self.data[idx:idx + self.window_size], self.data[idx + self.window_size]

    def __len__(self):
        return len(self.data) - self.window_size


if __name__ == '__main__':
    parsed = False

    ds_collection = None
    if parsed:
        ds_collection = TextCorpusDatasetCollection().load(data_dir, window_size)
    else:
        ds_collection = TextCorpusDatasetCollection().parse(data_dir, window_size)
    ds_train = ds_collection.train
    ds_valid = ds_collection.valid
    ds_test = ds_collection.test
