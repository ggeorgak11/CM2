import torch
import torch.nn as nn
from torchvision import models
from .conv_lstm import ConvLSTM

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class ResNetUnetBlock(nn.Module):
    def __init__(self, n_channel_in, n_class_out, with_lstm):
        super(ResNetUnetBlock, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up1 = convrelu(64 + 128, 128, 3, 1)
        self.conv_up0 = convrelu(64 + 128, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

        # Layers for waypoint coverage prediction
        # Formulated as a classification problem where each waypoint can be either uncovered (0), or covered (1)
        self.conv_coverage = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.Lin_cov1 = nn.Linear(72, 32)
        self.relu_cov1 = nn.ReLU()
        self.Lin_cov2 = nn.Linear(32, n_class_out*2)

        self.with_lstm = with_lstm
        ## Use ConvLSTM with single feature channel i.e. B x num_waypoints x 1 x H x W
        if self.with_lstm:
            self.lstm_waypoints = ConvLSTM(input_dim=1, 
                                           hidden_dim=1, 
                                           kernel_size=(1,1), 
                                           num_layers=3,
                                           batch_first=True, 
                                           bias=True, 
                                           return_all_layers=False)


    def forward(self, input):
        input = self.upsample(input) # when grid_dim=192
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)

        layer2 = self.layer2_1x1(layer2)
        x = self.upsample(layer2)

        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        if self.with_lstm:
            out, _ = self.lstm_waypoints(out.unsqueeze(2))
            out = out[0].squeeze(2)

        # Prediction of covered waypoints 
        # (binary indicator of which waypoints are before or after the agent's current position)
        x_cov = self.conv_coverage(x)
        x_cov = x_cov.view(x_cov.shape[0], -1)
        x_cov = self.Lin_cov1(x_cov)
        x_cov = self.relu_cov1(x_cov)
        out_cov = self.Lin_cov2(x_cov)
        out_cov = out_cov.view(x_cov.shape[0], -1, 2)

        return out, out_cov


class ResNetUNetGoalPred(nn.Module):
    def __init__(self, n_channel_in, n_class_out, with_lstm):
        super().__init__()

        self.unet1 = ResNetUnetBlock(n_channel_in=n_channel_in, n_class_out=n_class_out, with_lstm=with_lstm)

    def forward(self, input):

        out1 = self.unet1(input=input)

        return out1