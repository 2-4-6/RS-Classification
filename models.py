from torchvision import models
import torch
from torch import nn

# First attempt at a model. CNN LSTM using conv2d
class SpatiotemporalModel(nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda"):
        super(SpatiotemporalModel, self).__init__()

        self.spatial_encoder = SpatialEncoder(input_dim=input_dim)
        output_dim = self.spatial_encoder.output_dim

        self.temporal_encoder = TemporalEncoder(input_dim=output_dim, num_classes=num_classes)

        self.modelname = f"Conv3d_LSTM_{input_dim}222"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

    def forward(self, x):
        x = self.spatial_encoder(x)
        x = self.temporal_encoder(x)
        return x

class SpatialEncoder(torch.nn.Module):
    def __init__(self, input_dim=4):
        super(SpatialEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.output_dim = self.model[0].out_channels

    def forward(self, x):
        N, T, D, H, W = x.shape
        x = self.model(x.view(N * T, D, H, W))
        return x.view(N, T, x.shape[1]) 

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TemporalEncoder, self).__init__()

        self.model = nn.LSTM(input_dim, num_classes, batch_first=True)
    
    def forward(self, x):
        x, _ = self.model(x)
        return x[:, -1, :]

class CNNLSTM(torch.nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda"):
        super(CNNLSTM, self).__init__()

        self.modelname = f"Conv3d_LSTM_{input_dim}22222222"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Replace input dimensions with dimensions of X
        self.lstm = nn.LSTM(1024, num_classes, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
        self.to(device)

    def forward(self, x):
        N, T, D, H, W = x.shape
        # Reshape input for CNN
        x = x.view(N, D, T, H, W)
        # Apply CNN layers
        x = self.cnn(x)

        # Reshape CNN output for LSTM
        x = x.view(N, T, -1)

        x, _ = self.lstm(x)

        # Get the output of the last time step
        x = x[:, -1, :]

        return x
