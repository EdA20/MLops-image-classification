import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from dataloader import get_cifar10_data
from omegaconf import DictConfig
from tqdm import tqdm
from utils import instantiate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conf_dir = str(Path(__file__).resolve().parent.parent / "conf")


def val_epoch(model, criterion, loader):
    loss_log = []
    acc_list = []
    model.eval()

    for images, labels in tqdm(loader):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)

        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == labels).float().mean()
        acc_list.append(acc.item())

    return np.mean(loss_log), np.mean(acc_list)


def train_epoch(model, optimizer, criterion, train_loader):
    loss_log = []
    acc_log = []
    model.train()

    for images, labels in tqdm(train_loader):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size

        optimizer.zero_grad()
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == labels).float().mean()
        acc_log.append(acc.item())

    return loss_log, acc_log


@hydra.main(version_base=None, config_path=conf_dir, config_name="config")
def train(cfg: DictConfig):
    train_loader, val_loader = get_cifar10_data(cfg.dataloader, True)
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    model = instantiate(cfg.train.model.class_name)()
    n_epochs = cfg.train.n_epochs
    optimizer = instantiate(cfg.train.optimizer.class_name)(
        model.parameters(), **cfg.train.optimizer.optimizer_params
    )
    criterion = instantiate(cfg.train.criterion.class_name)()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_acc = val_epoch(model, criterion, val_loader)

        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

    torch.save(model, "model.pth")


def main():
    subprocess.run(
        [
            "dvc",
            "get",
            "https://github.com/EdA20/MLops-image-classification",
            "data/cifar-10-batches-py",
        ]
    )
    train()


if __name__ == "__main__":
    main()
