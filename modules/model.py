import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, resolution, num_classes=6):
        super().__init__()

        self.feature_learning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            # 64
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # 32
            nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1),
            # 32
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
            # 32
        )
        self.classification = nn.Sequential(
            nn.Linear(in_features=((resolution ** 2) // 4) * 32, out_features=num_classes)
        )

    def forward(self, data):
        feature_map = self.feature_learning(data)
        flatten_f_map = torch.flatten(feature_map, 1)
        logits = self.classification(flatten_f_map)
        pred_probab = nn.Softmax(dim=1)(logits)
        return logits, pred_probab
