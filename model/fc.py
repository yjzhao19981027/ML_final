import torch.nn as nn


class fc(nn.Module):
    def __init__(self, input_size):
        super(fc, self).__init__()
        self.class_fc = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.class_fc(x)

