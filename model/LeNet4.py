import torch
import torch.nn as nn
from model.fc import fc
from model.cnn import cnn


class LeNet4(nn.Module):

    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = cnn()
        self.conv2 = cnn()
        self.conv3 = cnn()
        self.conv4 = cnn()
        self.fc = fc(input_size=4 * 4 * 16)

    def forward(self, patch1, patch2, patch3, patch4):
        feature1 = self.conv1(patch1)
        feature1 = feature1.view(feature1.shape[0], -1)  # 2*2*16 -> 64
        feature2 = self.conv2(patch2)
        feature2 = feature2.view(feature2.shape[0], -1)  # 2*2*16 -> 64
        feature3 = self.conv3(patch3)
        feature3 = feature3.view(feature3.shape[0], -1)  # 2*2*16 -> 64
        feature4 = self.conv4(patch4)
        feature4 = feature4.view(feature4.shape[0], -1)    # 2*2*16 -> 64
        feature1 = torch.cat((feature1, feature2), dim=1)
        feature2 = torch.cat((feature3, feature4), dim=1)
        feature = torch.cat((feature1, feature2), dim=1)
        class_out = self.fc(feature)
        return class_out

