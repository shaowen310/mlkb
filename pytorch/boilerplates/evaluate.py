import torch


def evaluate(model, dataloader, criterion, device='cpu'):
    model.to(device)
    criterion.to(device)
    
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for elems, labels in dataloader:
            elems = elems.to(device)
            labels = labels.to(device)

            preds = model(elems)
            loss = criterion(preds, labels)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
