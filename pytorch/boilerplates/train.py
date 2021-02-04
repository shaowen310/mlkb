def train(model, iterator, optimizer, criterion):
    epoch_loss = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
