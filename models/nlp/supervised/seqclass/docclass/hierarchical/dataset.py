# Adapted from https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/dataset.py
import pickle

import numpy as np
import torch


class HANDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super().__init__()

        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word

        index_to_word = []
        with open(dict_path, 'rb') as f:
            index_to_word = pickle.load(f)
        self.index_to_word = index_to_word

        mapped_doc = []
        with open(data_path, 'rb') as f:
            mapped_doc = pickle.load(f)
        doc_encodes, labels = self._prepare(mapped_doc)

        self.doc_encodes = torch.from_numpy(doc_encodes)
        self.labels = torch.from_numpy(labels)

        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.doc_encodes[index], self.labels[index]

    def _prepare(self, mapped_doc):
        labels = []
        doc_encodes = []
        for (label, doc) in mapped_doc:
            labels.append(label)
            doc_encodes.append(self._pad(doc))
        return np.array(doc_encodes, dtype=np.int64), np.array(labels, dtype=np.int64)

    def _pad(self, document_encode, pad_idx=0):
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [pad_idx for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[pad_idx for _ in range(self.max_length_word)]
                                  for _ in range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word]
                           for sentences in document_encode][:self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)

        return document_encode.astype(np.int64)


if __name__ == '__main__':
    test = HANDataset(data_path="./data_/ag_news/train.pkl", dict_path="./data_/ag_news/i2w.pkl")
    print(test.__getitem__(index=1)[0].shape)
