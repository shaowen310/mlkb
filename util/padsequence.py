import numpy as np


def pad_sequence(sequences, batch_first=True, max_len=None, dtype=np.int, padding='post', truncating='post', value=0.):
    trailing_dims = sequences[0].shape[1:]
    if max_len == None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = np.full(out_dims, value, dtype)

    for i, tensor in enumerate(sequences):
        length = min((len(tensor), max_len))
        if batch_first:
            if padding == 'post' and truncating == 'post':
                out_tensor[i, :length, ...] = tensor[:length, ...]
            if padding == 'pre' and truncating == 'post':
                out_tensor[i, -length:, ...] = tensor[:length, ...]
            if padding == 'post' and truncating == 'pre':
                out_tensor[i, :length, ...] = tensor[-length:, ...]
            if padding == 'pre' and truncating == 'pre':
                out_tensor[i, -length:, ...] = tensor[-length:, ...]
        else:
            if padding == 'post' and truncating == 'post':
                out_tensor[:length, i, ...] = tensor[:length, ...]
            if padding == 'pre' and truncating == 'post':
                out_tensor[-length:, i, ...] = tensor[:length, ...]
            if padding == 'post' and truncating == 'pre':
                out_tensor[:length, i, ...] = tensor[-length:, ...]
            if padding == 'pre' and truncating == 'pre':
                out_tensor[-length:, i, ...] = tensor[-length:, ...]

    return out_tensor
