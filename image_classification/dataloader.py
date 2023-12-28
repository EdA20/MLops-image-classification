import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def get_cifar10_data(batch_size=64, train=True):
    torch.manual_seed(0)
    np.random.seed(0)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # Загружаем данные
    set = torchvision.datasets.CIFAR10(root="./", train=train, transform=transform)

    # В датасете определено разбиение только на train и test,
    # так что валидацию дополнительно выделяем из обучающей выборки
    if train:
        train_idx, valid_idx = train_test_split(
            np.arange(len(set)), test_size=0.3, shuffle=True, random_state=0
        )
        trainset = torch.utils.data.Subset(set, train_idx)
        valset = torch.utils.data.Subset(set, valid_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader

    else:
        test_loader = torch.utils.data.DataLoader(
            set, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return test_loader
