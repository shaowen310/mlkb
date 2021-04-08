'''
Cite:
    
'''
import os
import urllib


class SampleData:
    _DOWNLOAD_URL = ''
    DATASET_NAME = ''
    TRAIN_DATA_NAME = ''

    def __init__(self, root='data_'):
        self.data_dir = os.path.join(root, self.DATASET_NAME)

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.train_fp = os.path.join(self.data_dir, self.TRAIN_DATA_NAME)

        if not os.path.exists(self.train_fp):
            self._download()

    def _download(self):
        urllib.request.urlretrieve(self._DOWNLOAD_URL, self.train_fp)

    def _unzip(self):
        pass

    def generate_samples(self):
        data = {}
        yield data
