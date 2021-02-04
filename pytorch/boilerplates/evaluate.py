import torch


def evaluate(model, dataloader, criterion):
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for elems, labels in dataloader:
            preds = model(elems)
            loss = criterion(preds, labels)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
