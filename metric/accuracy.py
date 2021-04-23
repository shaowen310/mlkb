# %%
import numpy as np


def accuracy_score(truth, preds, normalize=True):
    predicted_labels = np.argmax(preds, dim=1)
    n_matches = np.sum(predicted_labels == truth)

    return n_matches / len(truth) if normalize else n_matches
