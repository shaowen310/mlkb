class BatchTextCorpusCollection:
    def __init__(self, corpus):
        self.root_dir = ''
        self.corpus = corpus
        self.train = None
        self.test = None
        self.valid = None

    def __init_datasets(self):
        self.train = BatchText(self.corpus.train, self.batch_size)
        self.test = BatchText(self.corpus.test, self.batch_size)
        self.valid = BatchText(self.corpus.valid, self.batch_size)

    def parse(self, root_dir, batch_size):
        '''
        file organization
        Train: root_dir/train.txt
        Test: root_dir/test.txt
        Validation: root_dir/valid.txt
        '''
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.corpus.parse(root_dir)
        self.__init_datasets()

    def save(self, root_dir):
        self.corpus.save(root_dir)

    def load(self, root_dir, batch_size):
        '''
        file organization
        Dictionary: root_dir/dict.pickle
        Train: root_dir/train.pt
        Test: root_dir/test.pt
        Validation: root_dir/valid.pt
        '''
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.corpus.load(root_dir)
        self.__init_datasets()


class BatchText:
    def __init__(self, data, batch_size):
        '''
        data of size (-1, batch_size)
        '''
        self.batch_size = batch_size
        # Work out how cleanly we can divide the dataset into window_size parts.
        n_batches = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, n_batches * batch_size)
        self.data = data.view(batch_size, -1).t().contiguous()

    def __len__(self):
        return len(self.data)

    def get(self, i, seq_len):
        seq_len_ = min(seq_len, len(self.data) - 1 - i)
        data = self.data[i:i + seq_len_]
        target = self.data[i + 1:i + 1 + seq_len_].view(-1)
        return data, target
