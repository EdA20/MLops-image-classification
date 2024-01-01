from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from utils import instantiate

conf_dir = str(Path(__file__).resolve().parent.parent / "conf")


@dataclass
class Cnst:
    classes: Tuple[str] = field(
        default_factory=lambda: (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    )


def get_cifar10_data(cfg: DictConfig, train: bool):
    torch.manual_seed(0)
    np.random.seed(0)

    transformations = []
    transforms_yaml = cfg.transforms
    transforms_params = cfg.transforms_params
    for name, transform in transforms_yaml.items():
        params = transforms_params.get(name)
        if params is not None:
            transform = instantiate(transform)(**params)
        else:
            transform = instantiate(transform)()
        transformations.append(transform)

    transform = transforms.Compose(transformations)
    # Загружаем данные
    set = torchvision.datasets.CIFAR10(root="./", train=train, transform=transform)

    # В датасете определено разбиение только на train и test,
    # так что валидацию дополнительно выделяем из обучающей выборки
    if train:
        train_idx, valid_idx = train_test_split(
            np.arange(len(set)), **cfg.train_test_split
        )
        trainset = torch.utils.data.Subset(set, train_idx)
        valset = torch.utils.data.Subset(set, valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, **cfg.dataloader)
        val_loader = torch.utils.data.DataLoader(valset, **cfg.dataloader)
        return train_loader, val_loader

    else:
        test_loader = torch.utils.data.DataLoader(set, **cfg.dataloader)
        return test_loader
