# %%
from torchtext.datasets import text_classification

# %%
# word-level encoded
datasets = [
    'AG_NEWS', 'SogouNews', 'DBpedia', 'YelpReviewPolarity', 'YelpReviewFull', 'YahooAnswers', 'AmazonReviewPolarity',
    'AmazonReviewFull'
]

ds_train, ds_test = text_classification.DATASETS['AG_NEWS']()

ds_train.get_vocab()
