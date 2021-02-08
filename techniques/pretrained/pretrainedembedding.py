import torch


def create_weight_matrix(pretrained_embeddings, embedding_word2idx, target_vocab):
    n_words = len(target_vocab)
    embedding_dim = pretrained_embeddings.size()[1]
    weight_matrix = torch.zeros((n_words, embedding_dim))
    words_not_found = []

    for i, word in enumerate(target_vocab):
        if word not in embedding_word2idx:
            words_not_found.append(word)
            weight_matrix[i] = torch.randn((embedding_dim, ))
            continue
        weight_matrix[i] = pretrained_embeddings[embedding_word2idx[word]]

    return weight_matrix, words_not_found


def create_emb_layer(weight_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weight_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weight_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
