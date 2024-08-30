import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetWithAttention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetWithAttention, self).__init__()
        # Encoder
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(128, 128, 64)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # Decoder
        x = self.up1(x3)
        x = self.att1(x, x2)
        x = self.up2(x)
        x = self.att2(x, x1)
        x = self.outc(x)
        return x

