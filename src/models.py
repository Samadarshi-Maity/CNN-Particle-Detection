# Class to load the dataset.
from torch import nn
import torch
import cv2
import numpy as np

# define the UNet Model
class UNetHeatmap(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        # Encoder 
        
        # ... layer 1 with 2 conv and 1 maxpool 
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        # ... layer 2 with 2 conv and 1 maxpool 
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck  shifting from 64 to 128 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )

        # Decoder
        # ... layer 3 with 2 conv
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        
        # ... layer 4 with 1 upconv and 2 conv  
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU()
        )

        # Output Layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    
    def forward(self, x):
        # Look at the comments carefully to see the dimension flow of the entire structure 
        #[batch, filters, height, width] 
        # Encoder path 
        enc1 = self.enc1(x)         # [B, 32, H, W]
        x = self.pool1(enc1)        # [B, 32, H/2, W/2]

        enc2 = self.enc2(x)         # [B, 64, H/2, W/2]
        x = self.pool2(enc2)        # [B, 64, H/4, W/4]

        # Bottleneck
        x = self.bottleneck(x)      # [B, 128, H/4, W/4]

        # Decoder path
        x = self.upconv2(x)         # [B, 64, H/2, W/2]
        x = torch.cat([x, enc2], dim=1)  # skip connection
        x = self.dec2(x)            # [B, 64, H/2, W/2]

        x = self.upconv1(x)         # [B, 32, H, W]
        x = torch.cat([x, enc1], dim=1)  # skip connection
        x = self.dec1(x)            # [B, 32, H, W]

        x = self.final_conv(x)      # [B, out_channels, H, W]
        return x # do not use a sigmoid here ... we will lose information here 
    
# define a ResNet model 
# ..... this is still on the experimental side...... is larger but shows a similar performance to the 

# ...... Build a residual block 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNetHeatmap(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.in_channels = 64

        # ......Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ...... Encoder
        self.layer1 = self._make_layer(64, 2, downsample=False)
        self.layer2 = self._make_layer(128, 2, downsample=True)
        self.layer3 = self._make_layer(256, 2, downsample=True)

        # ....... Decoder (Upsampling layers inline)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Final layer (no sigmoid)
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)

    # ......... Build the residual block
    def _make_layer(self, out_channels, blocks, downsample):
        layers = [ResidualBlock(self.in_channels, out_channels, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    # .......... connect the layers
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.out(x) 
        return x
