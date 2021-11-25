import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),     # 32*32*1 -> 28*28*6
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # 28*28*6 -> 14*14*6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),    # 14*14*6 -> 10*10*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # 10*10*16 -> 5*5*16(400)
        )
        self.class_fc = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.conv2(feature)
        feature = feature.view(feature.shape[0], -1)    # 5*5*16 -> 400
        class_out = self.class_fc(feature)
        return class_out

