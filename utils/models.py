from torchvision import models
import torch
from torch import nn

SPATIAL = 'Conv2d'
TEMPORAL = 'LSTM'

class SpatiotemporalModel(nn.Module):
    def __init__(self, spatial_backbone="mobilenet_v3_small", temporal_backbone="LSTM", input_dim=4,
                 num_classes=9, sequencelength=365, pretrained_spatial=True, device="cuda"):
        super(SpatiotemporalModel, self).__init__()

        self.spatial_encoder = SpatialEncoder(backbone=spatial_backbone , input_dim=input_dim, pretrained=pretrained_spatial)
        output_dim = self.spatial_encoder.output_dim

        self.temporal_encoder = TemporalEncoder(backbone=temporal_backbone, input_dim=output_dim, num_classes=num_classes, sequencelength=sequencelength, device=device)

        self.modelname = f"{spatial_backbone}_{temporal_backbone}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

    def forward(self, x):
        x = self.spatial_encoder(x)
        x = self.temporal_encoder(x)
        return x
    
class SpatialEncoder(torch.nn.Module):
    def __init__(self, backbone, input_dim=4, pretrained=False):
        super(SpatialEncoder, self).__init__()
        cnn = models.__dict__[backbone](pretrained=pretrained).features
        cnn[0][0] = nn.Conv2d(input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model = cnn
        self.output_dim = cnn[-1][0].out_channels

    def forward(self, x):
        N, T, D, H, W = x.shape
        x = self.model(x.view(N * T, D, H, W))
        return x.view(N, T, x.shape[1])

class TemporalEncoder(nn.Module):
    def __init__(self, backbone, input_dim, num_classes, sequencelength, device):
        super(TemporalEncoder, self).__init__()

        self.model = nn.LSTM(input_dim, num_classes, batch_first=True)
    
    def forward(self, x):
        x, _ = self.model(x)
        return x[:, -1, :] 