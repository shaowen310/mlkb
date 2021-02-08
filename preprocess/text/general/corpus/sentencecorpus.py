import os
import collections
import pickle
import torch


class Dictionary:
    def __init__(self, idx2word=None):
        if idx2word is None:
            self.word2idx = {}
            self.idx2word = []
        else:
            self.idx2word = idx2word
            self.word2idx = {word: idx for idx, word in enumerate(idx2word)}

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def save(self, fp):
        pickle.dump(self.idx2word, open(fp, 'wb'))

    def load(self, fp):
        self.idx2word = pickle.load(open(fp, 'rb'))
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}


class SentenceCorpus:
    def __init__(self, tokenizer):
        self.dict = Dictionary()
        self.tokenizer = tokenizer
        self.train = torch.tensor(())
        self.valid = torch.tensor(())
        self.test = torch.tensor(())

    def parse(self, path):
        self.train = self.tokenize(os.path.join(path, 'train.txt'), True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), False)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), False)

    def tokenize(self, path, train):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        sentences = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                sent = self.tokenizer.tokenize(line) + ['<eos>']
                sentences.append(sent)

        if train:
            wc = collections.Counter(word for sent in sentences for word in sent)
            vocab = [w for w, _ in wc.most_common()]
            self.dict = Dictionary(vocab)
            self.dict.add_word('<unk>')

        idss = []
        for sent in sentences:
            ids = []
            for word in sent:
                if word not in self.dict.word2idx:
                    word = '<unk>'
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
