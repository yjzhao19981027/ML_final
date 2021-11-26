import torch.nn as nn


class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),     # 32*32*3 -> 28*28*6    20*20*3 -> 16*16*6
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # 28*28*6 -> 14*14*6    16*16*6 -> 8*8*6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),    # 14*14*6 -> 10*10*16   8*8*6 -> 4*4*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # 10*10*16 -> 5*5*16(400)   4*4*16 -> 2*2*16
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x