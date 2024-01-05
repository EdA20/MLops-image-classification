import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import torch
from dataloader import Cnst, get_cifar10_data
from omegaconf import DictConfig
from tqdm import tqdm

date = datetime.today().strftime("%Y-%m-%d")

conf_dir = str(Path(__file__).resolve().parent.parent / "conf")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONSTANTS = Cnst()


@hydra.main(version_base=None, config_path=conf_dir, config_name="config_infer")
def main(cfg: DictConfig):
    model = torch.load("model.pth")
    loader = get_cifar10_data(cfg.dataloader, False)

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

    output_dir = Path(__file__).resolve().parent.parent / f"outputs/{date}"
    child = os.listdir(output_dir)[-1]
    file = str(output_dir / child / "predictions.csv")

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in preds:
            writer.writerow(row)

    subprocess.run(["dvc", "add", file])
    subprocess.run(["dvc", "push"])
    subprocess.run(["git", "add", f"{str(file) + '.dvc'}"])
    subprocess.run(
        ["git", "commit", "-m", f"'predictions.csv added ({date + ' ' + child})'"]
    )


if __name__ == "__main__":
    main()
