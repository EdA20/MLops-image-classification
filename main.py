from test import test

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from dataloader import get_cifar10_data
from model import BasicBlockNet
from train import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_loader, val_loader, test_loader = get_cifar10_data(
        batch_size=64, transform=transform
    )
    net = BasicBlockNet()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_log, train_acc_log, val_loss_log, val_acc_log = train(
        net, optimizer, criterion, 20, train_loader, val_loader
    )
    val_loss, val_acc = test(net, criterion, test_loader)
    print(f" test loss: {val_loss}, test acc: {val_acc}\n")
