import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import h5py
import sys
import time

class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return residual + out

class Cropping1D(nn.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]

class SpliceAI(nn.Module):
    def __init__(self, L, W, AR):
        super(SpliceAI, self).__init__()
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.residual_units = nn.ModuleList([ResidualUnit(L, W[i], AR[i]) for i in range(len(W))])
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))  # Adjust this based on your specific needs

    def forward(self, x):
        x = self.initial_conv(x)
        for ru in self.residual_units:
            x = ru(x)
        x = self.crop(x)  # Apply cropping here
        out = self.final_conv(x)
        return out
        # return F.softmax(out, dim=1)  # Consider returning logits during training
