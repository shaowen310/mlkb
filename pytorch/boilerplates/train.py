import time


def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device, log_interval=100):
    model.train()

    epoch_loss = 0
    log_loss = 0
    start_time = time.time()

    for batch, (elems, labels) in enumerate(dataloader):
        elems = elems.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(elems)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        log_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = log_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, elapsed * 1000 / log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            log_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)
