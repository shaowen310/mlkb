# %%
import torch
import torchtext
from torchtext.datasets import text_classification

import os

# %%
# word-level
datasets = [
    'AG_NEWS', 'SogouNews', 'DBpedia', 'YelpReviewPolarity', 'YelpReviewFull', 'YahooAnswers',
    'AmazonReviewPolarity', 'AmazonReviewFull'
]

ds_train, ds_test = text_classification.DATASETS['AG_NEWS']()

ds_train.get_vocab()

# %%
# string
_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


