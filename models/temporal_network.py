import torch
import torch.nn as nn

class TemporalConsistencyNet(nn.Module):
    def __init__(self):
        super(TemporalConsistencyNet, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3d_3 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv3d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = nn.ReLU()(x)
        x = self.conv3d_2(x)
        x = nn.ReLU()(x)
        x = self.conv3d_3(x)
        x = nn.ReLU()(x)
        x = self.final_conv(x)
        return x

