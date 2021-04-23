import math
import time
import torch


# %%



def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device, log_interval=1000):
    model.to(device)
    criterion.to(device)

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

        batch_loss = loss.item()
        epoch_loss += batch_loss
        log_loss += batch_loss

        if log_interval and batch % log_interval == 0 and batch > 0:
            cur_loss = log_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | {:5.2f} ms/batch  | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(epoch, batch, elapsed * 1000 / log_interval,
                                                        cur_loss, math.exp(cur_loss)))
            log_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)


if __name__ == '__main__':
    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(5):
        epoch_loss = 0
        print('| epoch {} | epoch_loss {} |'.format(epoch + 1, epoch_loss))
