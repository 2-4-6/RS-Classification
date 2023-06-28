from torchvision import models
import torch
from torch import nn

class SpatiotemporalModel(nn.Module):
    def __init__(self, input_dim=4, num_classes=9, device="cuda"):
        super(SpatiotemporalModel, self).__init__()

        self.spatial_encoder = SpatialEncoder(input_dim=input_dim)
        output_dim = self.spatial_encoder.output_dim

        self.temporal_encoder = TemporalEncoder(input_dim=output_dim, num_classes=num_classes)

        self.modelname = f"Conv2d_LSTM"

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
            nn.Conv2d(input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.output_dim = self.model[0].out_channels

    def forward(self, x):
        N, T, D, H, W = x.shape
        print(x.shape)
        x = self.model(x.view(N * T, D, H, W))
        print(x.shape)
        return x.view(N, T, x.shape[1])

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TemporalEncoder, self).__init__()

        self.model = nn.LSTM(input_dim, num_classes, batch_first=True)
    
    def forward(self, x):
        x, _ = self.model(x)
        return x[:, -1, :]