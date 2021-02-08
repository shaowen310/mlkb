import pickle


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __init__(self, idx2word):
        self.idx2word = idx2word
        self.word2idx = {word: idx for idx, word in enumerate(idx2word)}

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
