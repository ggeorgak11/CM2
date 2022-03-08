import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class MapEncoder(nn.Module):
    def __init__(self, n_channel_in, n_channel_out):    
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)

        self.conv_last = nn.Conv2d(128, n_channel_out, kernel_size=3, stride=2, padding=1, bias=False)


    def forward(self, input):

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        out = self.conv_last(layer2)
        
        return out