import torch


def create_weight_matrix(pretrained_embedding, embedding_word2idx, target_vocab):
    n_words = len(target_vocab)
    embedding_dim = pretrained_embedding.size()[1]
    weight_matrix = torch.zeros((n_words, embedding_dim))
    words_not_found = []

    for i, word in enumerate(target_vocab):
        if word not in embedding_word2idx:
            words_not_found.append(word)
            weight_matrix[i] = torch.randn((embedding_dim, ))
            continue
        weight_matrix[i] = pretrained_embedding[embedding_word2idx[word]]

    return weight_matrix, words_not_found


def create_embedding_layer(pretrained_embedding,
                           embedding_word2idx,
                           target_vocab,
                           non_trainable=False):
    weight_matrix, words_not_found = create_weight_matrix(pretrained_embedding, embedding_word2idx,
                                                          target_vocab)
    n_embeddings, embedding_dim = weight_matrix.size()
    emb_layer = torch.nn.Embedding(n_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weight_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, words_not_found
