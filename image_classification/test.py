import csv
from pathlib import Path

import hydra
import torch
from dataloader import Cnst, get_cifar10_data
from omegaconf import DictConfig
from tqdm import tqdm

conf_dir = str(Path(__file__).resolve().parent.parent / "conf/dataloader")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONSTANTS = Cnst()


@hydra.main(version_base=None, config_path=conf_dir, config_name="dataloader_cfg")
def main(cfg: DictConfig):
    model = torch.load("model.pth")
    loader = get_cifar10_data(cfg, False)

    model.eval()

    preds = [["predictions"]]
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(
                device
            )  # images: batch_size x num_channels x height x width
            logits = model(images)  # logits: batch_size x num_classes

            pred = logits.argmax(dim=1).tolist()
            pred = [[CONSTANTS.classes[i]] for i in pred]
            preds.extend(pred)

    with open("preds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for row in preds:
            writer.writerow(row)


if __name__ == "__main__":
    main()
