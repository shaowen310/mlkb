'''
Cite:
    @inproceedings{Zhang2015CharacterlevelCN,
        title={Character-level Convolutional Networks for Text Classification},
        author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
        booktitle={NIPS},
        year={2015}
    }
'''
import os
import urllib
import csv


class AGNewsData:
    _TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    _TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    DATASET_NAME = 'ag_news'
    TRAIN_DATA_NAME = 'train.csv'
    TEST_DATA_NAME = 'test.csv'

    def __init__(self, root='data_'):
        self.data_dir = os.path.join(root, self.DATASET_NAME)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.train_file = os.path.join(self.data_dir, self.TRAIN_DATA_NAME)
        self.test_file = os.path.join(self.data_dir, self.TEST_DATA_NAME)

        if not os.path.exists(self.train_file):
            self._download()

    def _download(self):
        urllib.request.urlretrieve(self._TRAIN_DOWNLOAD_URL, self.train_file)
        urllib.request.urlretrieve(self._TEST_DOWNLOAD_URL, self.test_file)

    def generate_samples(self, filepath):
        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for id_, row in enumerate(csv_reader):
                label, title, description = row
                # Original labels are [1, 2, 3, 4] ->
                #                   ['World', 'Sports', 'Business', 'Sci/Tech']
                # Re-map to [0, 1, 2, 3].
                label = int(label) - 1
                text = ". ".join((title, description))
                yield id_, {"text": text, "label": label}
