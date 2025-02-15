import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, 
                 base_channels=64, 
                 strides=(2, 2, 2, 2), 
                 kernel_size=3, 
                 activation=nn.ReLU):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        prev_channels = in_channels
        for stride in strides:
            self.encoder.append(self.conv_block(prev_channels, 
                                                base_channels, 
                                                kernel_size, 
                                                activation))
            self.encoder.append(nn.MaxPool2d(kernel_size=2, 
                                                stride=stride))
            prev_channels = base_channels
            base_channels *= 2
        
        self.bottleneck = self.conv_block(prev_channels, 
                                            base_channels, 
                                            kernel_size, 
                                            activation)
        
        for stride in reversed(strides):
            base_channels //= 2
            self.decoder.append(nn.ConvTranspose2d(prev_channels, 
                                                    base_channels, 
                                                    kernel_size=2, 
                                                    stride=stride))
            self.decoder.append(self.conv_block(prev_channels, 
                                                base_channels, 
                                                kernel_size, 
                                                activation))
            prev_channels = base_channels
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, kernel_size, activation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            activation()
        )
        
    def forward(self, x):
        enc_features = []
        for enc_layer in self.encoder:
            x = enc_layer(x)
            if isinstance(enc_layer, nn.Sequential):
                enc_features.append(x)
        
        x = self.bottleneck(x)
        
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            enc_feat = enc_features.pop()
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder[i + 1](x)
        
        return self.final_conv(x)
