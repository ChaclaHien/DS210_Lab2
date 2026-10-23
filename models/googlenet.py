import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, c1, c2, c3, c4):
        # c1: 1x1; c2: (1x1→3x3); c3: (1x1→5x5); c4: (3x3pool→1x1)
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv2d(in_ch, c1, 1), nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_ch, c2[0], 1), nn.ReLU(),
                                nn.Conv2d(c2[0], c2[1], 3, padding=1), nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_ch, c3[0], 1), nn.ReLU(),
                                nn.Conv2d(c3[0], c3[1], 5, padding=2), nn.ReLU())
        self.p4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
                                nn.Conv2d(in_ch, c4, 1), nn.ReLU())
    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # padding=3 cho conv đầu; ceil_mode=True cho tất cả MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )
        self.b3 = nn.Sequential(
            InceptionBlock(192, 64,  (96,128),  (16,32),  32),
            InceptionBlock(256, 128, (128,192), (32,96),  64),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )
        self.b4 = nn.Sequential(
            InceptionBlock(480, 192, (96,208),  (16,48),  64),
            InceptionBlock(512, 160, (112,224), (24,64),  64),
            InceptionBlock(512, 128, (128,256), (24,64),  64),
            InceptionBlock(512, 112, (144,288), (32,64),  64),
            InceptionBlock(528, 256, (160,320), (32,128), 128),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )
        self.b5 = nn.Sequential(
            InceptionBlock(832, 256, (160,320), (32,128), 128),
            InceptionBlock(832, 384, (192,384), (48,128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.b5(self.b4(self.b3(self.stem(x)))))
