import torch


def evaluate(model, iterator, criterion):
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
