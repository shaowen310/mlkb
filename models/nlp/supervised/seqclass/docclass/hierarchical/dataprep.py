# Adapted from https://github.com/tqtg/hierarchical-attention-networks/blob/master/data_prepare.py
import os
import nltk
import itertools
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
from sklearn.model_selection import train_test_split

from agnews import AGNewsData

# Hyper parameters
WORD_CUT_OFF = 5


def build_vocab(docs, save_path):
    print('Building vocab ...')

    sents = itertools.chain.from_iterable([sent_tokenize(doc) for doc in docs])
    tokenized_sents = [word_tokenize(sent) for sent in sents]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
    print("%d unique words found" % len(word_freq.items()))

    # Cut-off
    retained_words = [w for (w, f) in word_freq.items() if f > WORD_CUT_OFF]
    print("%d words retained" % len(retained_words))

    # Get the most common words and build index_to_word and word_to_index vectors
    # Word index starts from 2, 1 is reserved for UNK, 0 is reserved for padding
    word_to_index = {'PAD': 0, 'UNK': 1}
    for i, w in enumerate(retained_words):
        word_to_index[w] = i + 2
    index_to_word = {i: w for (w, i) in word_to_index.items()}

    print("Vocabulary size = %d" % len(word_to_index))

    with open(os.path.join(save_path, 'w2i.pkl'), 'wb') as f:
        pickle.dump(word_to_index, f)

    with open(os.path.join(save_path, 'i2w.pkl'), 'wb') as f:
        pickle.dump(index_to_word, f)

    return word_to_index


def process(word_to_index, data):
    mapped_data = []
    for label, doc in data:
        mapped_doc = [[word_to_index.get(word, 1) for word in sent.split()] for sent in doc.split('<sssss>')]
        mapped_data.append((label, mapped_doc))
    return mapped_data


def save_processed(mapped_data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(mapped_data, f)


def read_data(data_file):
    data = pd.read_csv(data_file, sep='\t', header=None, usecols=[4, 6])
    print('{}, shape={}'.format(data_file, data.shape))
    return data


if __name__ == '__main__':
    agnewsdata = AGNewsData()

    d_train_stream = agnewsdata.generate_samples(agnewsdata.train_file)
    d_train_doc_stream = map(lambda d: d[1]['text'], d_train_stream)
    word_to_index = build_vocab(d_train_doc_stream, os.path.join(agnewsdata.data_dir, ))
    d_train_stream = agnewsdata.generate_samples(agnewsdata.train_file)
    d_train_label_doc_stream = map(lambda d: (d[1]['label'], d[1]['text']), d_train_stream)
    mapped_train = process(word_to_index, d_train_label_doc_stream)

    labels = [label for (label, _) in mapped_train]
    (mapped_train, mapped_dev) = train_test_split(mapped_train, test_size=0.1, stratify=labels)

    save_processed(mapped_train, os.path.join(agnewsdata.data_dir, 'train.pkl'))
    save_processed(mapped_dev, os.path.join(agnewsdata.data_dir, 'dev.pkl'))

    d_test_stream = agnewsdata.generate_samples(agnewsdata.test_file)
    d_test_label_doc_stream = map(lambda d: (d[1]['label'], d[1]['text']), d_test_stream)
    mapped_test = process(word_to_index, d_test_label_doc_stream)
    save_processed(mapped_test, os.path.join(agnewsdata.data_dir, 'test.pkl'))
