import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2 if in_ch != out_ch else 1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False), nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        # MaxPool giữa các block (k=3, s=2, p=0)
        self.layer1 = nn.Sequential(ResidualBlock(64,  64),  ResidualBlock(64,  64))
        self.pool1  = nn.MaxPool2d(3, stride=2, padding=0)
        self.layer2 = nn.Sequential(ResidualBlock(64,  128), ResidualBlock(128, 128))
        self.pool2  = nn.MaxPool2d(3, stride=2, padding=0)
        self.layer3 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256))
        self.pool3  = nn.MaxPool2d(3, stride=2, padding=0)
        self.layer4 = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return self.head(self.layer4(x))
