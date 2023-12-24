import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, criterion, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for images, labels in tqdm(loader):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)

        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == labels).float().mean()
        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)
