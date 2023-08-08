from torchvision import models
import torch
from torch import nn
import random
from datetime import datetime
import time

class Conv3D_LSTM(torch.nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda", dropout_rate=0.5, model_name=None):
        super(Conv3D_LSTM, self).__init__()

        if model_name == None:
            model_name = time.strftime("%Y%m%d-%H%M%S")

        self.modelname = f"Conv3d_LSTM_{input_dim}_{model_name}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            # nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(4096, 100,num_layers=2, batch_first=True)

        self.fc = nn.Linear(100, num_classes)

        self.to(device)

    def forward(self, x):
        N, T, D, H, W = x.shape
        # input for CNN
        x = x.view(N, D, T, H, W)

        x = self.cnn(x)
        # print(x.shape)
        # shape CNN output for LSTM
        x = x.view(N, T, -1)
        # print(x.shape)

        x, _ = self.lstm(x)
        
        #output of the last time step
        x = x[:, -1, :]
        # print(x.shape)
        x = self.fc(x)

        return x