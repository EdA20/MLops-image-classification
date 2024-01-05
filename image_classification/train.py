import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from dataloader import get_cifar10_data
from omegaconf import DictConfig
from sklearn.metrics import precision_score
from tqdm import tqdm
from utils import instantiate, plot_losses

date = datetime.today().strftime("%Y-%m-%d")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conf_dir = str(Path(__file__).resolve().parent.parent / "conf")
log = logging.getLogger("train.py")


def download_data():
    subprocess.run(
        [
            "dvc",
            "get",
            "https://github.com/EdA20/MLops-image-classification",
            "data/cifar-10-batches-py",
        ]
    )


def upload_log_file():
    log_dir = Path(__file__).resolve().parent.parent / f"outputs/{date}"
    child = os.listdir(log_dir)[-1]
    file = log_dir / child / "train.log"
    subprocess.run(["dvc", "add", file])
    subprocess.run(["dvc", "push"])
    subprocess.run(["git", "add", f"{str(file) + '.dvc'}"])
    subprocess.run(["git", "commit", "-m", f"'train log added ({date + ' ' + child})'"])


def val_epoch(model, criterion, loader):
    loss_log, acc_log, precision_log = [], [], []
    model.eval()

    for images, labels in tqdm(loader):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size
        logits = model(images)  # logits: batch_size x num_classes

        loss = criterion(logits, labels)
        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == labels).float().mean()
        acc_log.append(acc.item())

        precision = precision_score(
            labels.tolist(), logits.argmax(dim=1).tolist(), average="micro"
        )
        precision_log.append(precision)

    return np.mean(loss_log), np.mean(acc_log), np.mean(precision_log)


def train_epoch(model, optimizer, criterion, train_loader):
    loss_log, acc_log, precision_log = [], [], []
    model.train()

    for images, labels in tqdm(train_loader):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size
        logits = model(images)  # logits: batch_size x num_classes

        loss = criterion(logits, labels)
        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == labels).float().mean()
        acc_log.append(acc.item())

        precision = precision_score(
            labels.tolist(), logits.argmax(dim=1).tolist(), average="micro"
        )
        precision_log.append(precision)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(loss_log), np.mean(acc_log), np.mean(precision_log)


@hydra.main(version_base=None, config_path=conf_dir, config_name="config")
def train(cfg: DictConfig):
    train_loader, val_loader = get_cifar10_data(cfg.dataloader, True)
    train_loss_log, train_acc_log, train_precision_log = [], [], []
    val_loss_log, val_acc_log, val_precision_log = [], [], []

    model = instantiate(cfg.train.model)()
    n_epochs = cfg.train.n_epochs
    optimizer = instantiate(cfg.train.optimizer)(
        model.parameters(), **cfg.train.optimizer_params
    )
    criterion = instantiate(cfg.train.criterion)()
    params, cfg_dataloader = dict(cfg.train), dict(cfg.dataloader)
    params.update(cfg_dataloader)

    for epoch in range(n_epochs):
        train_loss, train_acc, train_precision = train_epoch(
            model, optimizer, criterion, train_loader
        )
        val_loss, val_acc, val_precision = val_epoch(model, criterion, val_loader)

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)
        train_precision_log.append(train_precision)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)
        val_precision_log.append(val_precision)

        epoch_logging = f"Epoch {epoch+1}"
        train_loss_logging = f"train loss: {train_loss}, train acc: {train_acc}, train preicision: {train_precision}"
        val_loss_logging = f"val loss: {val_loss}, val acc: {val_acc}, val preicision: {val_precision}\n"

        log.info(epoch_logging)
        log.info(train_loss_logging)
        log.info(val_loss_logging)

    logging.shutdown()
    fig, file_path = plot_losses(
        train_loss_log,
        val_loss_log,
        train_acc_log,
        val_acc_log,
        train_precision_log,
        val_precision_log,
    )

    torch.save(model, "model.pth")

    mlflow.set_tracking_uri("http://127.0.1.1:8080")
    mlflow.set_experiment("Loss & metrics visualization")

    with mlflow.start_run(run_name="image_classification"):
        # Log the hyperparameters
        mlflow.log_params(params)
        mlflow.log_figure(fig, file_path)

        # Log the loss metric
        mlflow.log_metric("accuracy", train_acc_log[-1])

        mlflow.set_tag("Architecture", "ResNet")

        # Log the model
        mlflow.pytorch.log_model(model, "model")


def main():
    download_data()
    train()
    upload_log_file()


if __name__ == "__main__":
    main()
