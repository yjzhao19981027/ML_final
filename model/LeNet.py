import torch.nn as nn
from model.fc import fc
from model.cnn import cnn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = cnn()
        self.fc = fc(input_size=5 * 5 * 16)

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(feature.shape[0], -1)    # 5*5*16 -> 400
        class_out = self.fc(feature)
        return class_out

