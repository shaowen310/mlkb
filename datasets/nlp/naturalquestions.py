'''
Link:
    https://ai.google.com/research/NaturalQuestions/download
Cite:
@article{47761,
    title    = {Natural Questions: a Benchmark for Question Answering Research},
    author    = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
    year    = {2019},
    journal    = {Transactions of the Association of Computational Linguistics}
}
'''
import os
import gzip
import shutil
import json


class NaturalQuestionsData:
    _DOWNLOAD_URL_TRAIN = 'https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz'
    _DOWNLOAD_URL_DEV = 'https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz'

    DATASET_NAME = 'natural_questions'
    TRAIN_RAW_DATA_NAME = 'v1.0-simplified_simplified-nq-train.jsonl.gz'
    DEV_RAW_DATA_NAME = 'v1.0-simplified_nq-dev-all.jsonl.gz'
    TRAIN_DATA_NAME = 'simplified-nq-train.jsonl'
    DEV_DATA_NAME = 'nq-dev-all.jsonl'

    def __init__(self, root='data_'):
        self.data_dir = os.path.join(root, self.DATASET_NAME)

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.train_raw_fp = os.path.join(self.data_dir, self.TRAIN_RAW_DATA_NAME)
        self.train_fp = os.path.join(self.data_dir, self.TRAIN_DATA_NAME)
        self.dev_raw_fp = os.path.join(self.data_dir, self.DEV_RAW_DATA_NAME)
        self.dev_fp = os.path.join(self.data_dir, self.DEV_DATA_NAME)

        if not os.path.exists(self.train_raw_fp):
            self._download()

        if not os.path.exists(self.train_fp):
            self._unzip()

    def _download(self):
        # needs sign-in
        pass

    def _unzip(self):
        with gzip.open(self.train_raw_fp, 'rb') as f_in:
            with open(self.train_fp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        with gzip.open(self.dev_raw_fp, 'rb') as f_in:
            with open(self.dev_fp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def generate_samples(self, ds='train'):
        if ds=='train':
            fp = self.train_fp
        elif ds=='dev':
            fp = self.dev_fp
        else:
            raise ValueError()
        with open(fp, 'r') as f:
            for l in f:
                yield json.loads(l)
