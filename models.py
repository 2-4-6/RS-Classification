from torchvision import models
import torch
from torch import nn
import random
from datetime import datetime
import time

#Conv2d attempt
class Conv2D_x1_LSTM(torch.nn.Module):
    def __init__(self, input_dim, num_classes, device='cuda', test=False):
        super(Conv2D_x1_LSTM, self).__init__()

        if test:
            random_number = random.random()
        
        self.modelname = f"Conv2D_x1_LSTM_{input_dim}_bands_{random_number}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Replace input dimensions with dimensions of X
        self.lstm = nn.LSTM(96, num_classes, batch_first=True)

        self.to(device)

    def forward(self, x):
        N, T, D, H, W = x.shape
        # print(x.shape)
        x = (x.view(N * T, D, H, W))
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        x = x.view(N, T, -1)
        # print(x.shape)
        x, _ = self.lstm(x)

        x = x[:, -1, :]

        return x

class Conv3D_x1_LSTM(torch.nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda", dropout_rate=0.5, model_name=None):
        super(Conv3D_x1_LSTM, self).__init__()

        if model_name == None:
            model_name = time.strftime("%Y%m%d-%H%M%S")

        self.modelname = f"Conv3d_LSTM_{input_dim}_{model_name}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(18, num_classes)

        self.to(device)

    def forward(self, x):
        N, T, D, H, W = x.shape
        # input for CNN
        x = x.view(N, D, T, H, W)

        x = self.cnn(x)

        # Reshape CNN output for LSTM
        x = x.view(N, T, -1)
        x = self.dropout(x)

        self.lstm = nn.LSTM(x.shape[2], 18, batch_first=True).to(x.device)
        x, _ = self.lstm(x)

        #output of the last time step
        x = x[:, -1, :]

        x = self.fc(x)

        return x

class Conv3D_x2_LSTM(torch.nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda", dropout_rate=0.5, model_name=None):
        super(Conv3D_x2_LSTM, self).__init__()

        if model_name == None:
            model_name = datetime.now()

        self.modelname = f"Conv3d_X2_LSTM_{input_dim}_{model_name}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv3d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(18, num_classes)

        self.to(device)

    def forward(self, x):
        N, T, D, H, W = x.shape
        # input for CNN
        x = x.view(N, D, T, H, W)

        x = self.cnn(x)

        # Reshape CNN output for LSTM
        x = x.view(N, T, -1)
        x = self.dropout(x)

        self.lstm = nn.LSTM(x.shape[2], 18, batch_first=True).to(x.device)
        x, _ = self.lstm(x)

        #output of the last time step
        x = x[:, -1, :]

        x = self.fc(x)

        return x

# Varying number of layers?

class Conv3D_x3_LSTM(torch.nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda", test=False):
        super(Conv3D_x3_LSTM, self).__init__()

        if test:
            random_number = random.random()

        self.modelname = f"Conv3d_x3_LSTM_{input_dim}_bands_{random_number}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))


        self.cnn = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Replace input dimensions with dimensions of X
        self.lstm = nn.LSTM(64, num_classes, batch_first=True)

        self.fc = nn.Linear(num_classes, num_classes)

        self.to(device)

    def forward(self, x):
        N, T, D, H, W = x.shape
        x = x.view(N, D, T, H, W)

        x = self.cnn(x)

        x = x.view(N, T, -1)
        # print(x.shape)

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x
