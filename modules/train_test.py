import torch
from torch.autograd import Variable


def train(dataloader, model, loss_fn, optimizer, device, train_count, epoch_n):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    torch.manual_seed(epoch_n)
    for i, (imgs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        if torch.cuda.is_available():
            imgs = Variable(imgs.cuda())
            labels = Variable(labels.cuda())

        pred, _ = model(imgs)
        loss = loss_fn(pred, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * imgs.size(0)
        _, prediction = torch.max(pred.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_loss /= train_count
    train_accuracy /= train_count
    return train_accuracy, train_loss


def test(dataloader, model, device, test_count, loss_fn):
    model.eval()
    test_accuracy = 0.0
    test_loss = 0.0
    torch.manual_seed(2)
    for i, (imgs, labels) in enumerate(dataloader):
        if torch.cuda.is_available():
            imgs = Variable(imgs.cuda())
            labels = Variable(labels.cuda())

        pred, _ = model(imgs)
        loss = loss_fn(pred, labels)

        _, prediction = torch.max(pred.data, 1)

        test_loss += loss.item() * imgs.size(0)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy /= test_count
    test_loss /= test_count
    return test_accuracy, test_loss