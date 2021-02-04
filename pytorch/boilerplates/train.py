def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0

    model.train()

    for elems, labels in dataloader:
        elems = elems.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(elems).squeeze(1)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
