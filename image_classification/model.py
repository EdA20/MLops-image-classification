import torch.nn as nn


class BasicBlockNet(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
        )

        self.pooling = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=8))

        self.linear = nn.Linear(in_features=512, out_features=10)
        self.conv1x1 = nn.Conv2d(3, 32, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.conv1x1(x)
        out = self.encoder(x)
        out += identity
        out = self.pooling(out)
        out = self.linear(out.flatten(1))

        return out
