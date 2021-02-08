# https://github.com/pytorch/examples/tree/master/word_language_model/
from io import open
import torch


class TextCorpusDatasetCollection:
    def __init__(self, corpus):
        self.root_dir = ''
        self.corpus = corpus
        self.train = None
        self.test = None
        self.valid = None

    def __init_datasets(self):
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
        self.corpus.parse(root_dir)
        self.__init_datasets()

    def save(self, root_dir):
        self.corpus.save(root_dir)

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
        self.__init_datasets()


class TextCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, text_ids, window_size):
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

    ds_collection = None
    if parsed:
        ds_collection = TextCorpusDatasetCollection(corpus)
        ds_collection.load(data_dir, window_size)
    else:
        ds_collection = TextCorpusDatasetCollection(corpus)
        ds_collection.parse(data_dir, window_size)
        ds_collection.save(data_dir)
    ds_train = ds_collection.train
    ds_valid = ds_collection.valid
    ds_test = ds_collection.test
