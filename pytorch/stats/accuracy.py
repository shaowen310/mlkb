# %%
import torch


def accuracy_score(truth, preds, normalize=True):
    with torch.no_grad():
        predicted_labels = torch.argmax(preds, dim=1)
        n_matches = torch.sum(predicted_labels == truth)

        return n_matches.float() / len(truth) if normalize else n_matches
