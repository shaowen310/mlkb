# https://nlp.stanford.edu/projects/glove/
# Words are in lower case
import torch


def load_embedding(fp):
    embeddings = []
    embedding_word2idx = {}
    embedding_idx2word = []
    with open(fp, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            embedding_word2idx[word] = len(embedding_idx2word)
            embedding_idx2word.append(word)
            vect = torch.tensor(list(map(float, line[1:]))).float()
            embeddings.append(vect)
    embedding_matrix = torch.stack(embeddings, dim=0)
    return embedding_matrix, embedding_word2idx, embedding_idx2word
