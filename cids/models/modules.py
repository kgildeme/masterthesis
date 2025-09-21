import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_ReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(Conv2d_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return F.relu(self.conv(x))


class ResidualBlock2D(nn.Module):

    def __init__(self, in_channels, width=None):
        super(ResidualBlock2D, self).__init__()

        if width is None:
            width = in_channels
        
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(width, in_channels, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.branch(x) + x


class Conv1D_ReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(Conv1D_ReLU, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return F.relu(self.conv(x))
    

class ConvTranspose1D_ReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ConvTranspose1D_ReLU, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return F.relu(self.conv(x))
    

class ResidualBlock1D(nn.Module):
    
    def __init__(self, in_channels, width=None):
        super(ResidualBlock1D, self).__init__()

        if width is None:
            width = in_channels
        
        self.branch = nn.Sequential(
            nn.Conv1d(in_channels, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, in_channels, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.branch(x) + x