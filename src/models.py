# Load the necessary modules.
import torch
from torch import nn

# ................... define the UNet Model ........................
class UNetHeatmap(nn.Module):
    """
    UNet model for center detection via heatmap regression for overlapping particles.
    Implements 2 encoder layers, a bottleneck, 2 decoder layers, and 1 output layer.
    """
    def __init__(self, in_channels=1, out_channels=1):
        """
        Defines the constructor for designing the Unet model 

        Params: 
            in_channels  (int): number of channels in the image   
            out_channels (int): number of output channels in the heatmap
        """
        # define the instance data  channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize parent class to set up inherited attributes. 
        # Necessary for parameter tracking 
        super().__init__()
        
        # Encoder section 
        # ... layer 1 with 2 conv and 1 maxpool 
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        # ... layer 2 with 2 conv. and 1 maxpool layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck shifting from 64 to 128 channels 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )

        # Decoder 
        # 1st set of upsampling ... layer 3 with 1 upconv. and  2 conv. layer
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        
        # 2nd set of upsampling ... layer 4 with 1 upconv. and 2 conv. layer  
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU()
        )

        # Output Layer
        self.final_conv = nn.Conv2d(32, self.out_channels, kernel_size=1)

    # define the forward function 
    def forward(self, x):
        """
        Compute the forward pass of the U-Net. Check the inline comments below for the dimension flow
        Params:
            x (torch.Tensor): Input image tensor of shape [B,in_channels, H, W].
    
        Returns:
            x (torch.Tensor): Segmentation map of shape [B,out_channels, H, W].
        """
        # Encoder section 
        enc1 = self.enc1(x)         # [B, 32, H, W]
        x = self.pool1(enc1)        # [B, 32, H/2, W/2]

        enc2 = self.enc2(x)         # [B, 64, H/2, W/2]
        x = self.pool2(enc2)        # [B, 64, H/4, W/4]

        # Bottleneck
        x = self.bottleneck(x)      # [B, 128, H/4, W/4]

        # Decoder section
        x = self.upconv2(x)         # [B, 64, H/2, W/2]
        x = torch.cat([x, enc2], dim=1)  # [B, 128, H, W]  skip connection with channel concat.
        x = self.dec2(x)            # [B, 64, H/2, W/2]

        x = self.upconv1(x)         # [B, 32, H, W]
        x = torch.cat([x, enc1], dim=1)  # [B, 64, H, W] skip connection with channel concat.
        x = self.dec1(x)            # [B, 32, H, W]

        x = self.final_conv(x)      # [B, out_channels, H, W]
        return x                    # do not use a sigmoid here, or we will lose information here 



# ................................... ResNet model .................................................... 
# The model largely follows the UNet structure without having any global skip connections or bottlenecks. 
# Rapid downsampling at stem, the encoder is the residual blocks (3 layers with 6 blocks), and the decoder is 4 layers. 
# 
class ResidualBlock(nn.Module):
    """
    Defines a standard residual block. 
    This block has conv. layer,  batchnorm, ReLU repeated twice with a skip connection before the last ReLU
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        """
        Define the constructor for the residual block
        Params: 
            in_channels  (int): number of channels in the image   
            out_channels (int): number of output channels in the heatmap
        """
        # Initialize parent class to set up inherited attributes.
        # Necessary for parameter tracking 
        super().__init__()

        # define the instance data  channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Set the stride value based on the downsample boolean value 
        stride = 2 if downsample else 1

        # 1st set of conv., batchnorm and ReLU layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2nd set conv., batchnorm layers, the skip connection is added, and then ReLU is added
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # define the skip connections 
        # if channel sizes match, we sum identity; otherwise, use a conv. layer to match the channels. 
        self.skip = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    # Define the forward function for the residual block 
    def forward(self, x):
        """
        We connect the two sets of layers and the skip connection 

        Params:
            x (torch.Tensor): Input image tensor of shape [B,in_channels, H, W].
        Returns:
            x (torch.Tensor): Output image tensor of shape [B,in_channels, H, W].
        """
        # store the identity for the skip connection 
        identity = self.skip(x)

        # Apply the first set of layers
        out = self.relu(self.bn1(self.conv1(x)))

        # Apply the second set of layers 
        out = self.bn2(self.conv2(out))

        # Skip connection with elementwise sum 
        out += identity

        # returns the output with relu-based activation 
        return self.relu(out)

# Develop the full ResNet model with an encoding-decoding architecture
class ResNetHeatmap(nn.Module):
    def __init__(self, in_channels = 1, out_channels=1):
        """
        Creates the complete encoding-decoding architecture
        Implements 1 stem layer, 3 layers with 2 residual blocks each, and 4 upconv. layers.

        Params:
            in_channels  (int): number of channels in the image   
            out_channels (int): number of output channels in the heatmap
            
        """
        # Initialize parent class to set up inherited attributes.
        super().__init__()

        # set the data channels
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        
        #  Define the Stem layer, which contains a conv., batch norm, and ReLU layers with heavy downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Imp! sets the very first input to the residual layer series (private)
        self._res_channels = 64
        
        # 3 layers of residual blocks (2 in each layer) serve as the Encoder
        self.layer1 = self._make_layer(64, 2, downsample=False)
        self.layer2 = self._make_layer(128, 2, downsample=True)
        self.layer3 = self._make_layer(256, 2, downsample=True)

        # Decoder (Upsampling layers inline)

        # 1st set of upsampling ... 1 upconv. layer with batchnorm and ReLU
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 2nd set of upsampling ... 1 upconv. layer with batchnorm and ReLU
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 3rd set of upsampling ... 1 upconv. layer with batchnorm and ReLU
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 4th set of upsampling ... 1 upconv. layer with batchnorm and ReLU
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Final layer (no sigmoid) 1 conv. layer to resize the channels
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)

    # Build the residual block
    def _make_layer(self, out_channels, blocks, downsample):
        """
        Creates a residual layer with variable residual blocks (Basic blocks).  
        Sets the input channels of a block based on the output channel of the former block
        The input channel of the first block is set explicitly
        Params: 
            out_channels(int): the number of output channels 
            blocks(int)      : the number of blocks 
            downsample(bool) : stride set to 2 if true, else 1
        """     

        # creates the first residual block and add it to the list
        layers = [ResidualBlock(self._res_channels, out_channels, downsample)]

        # update the input channel for the next block with the output channel 
        self._res_channels = out_channels

        # develop a for loop for dynamic layer creation based on the desired number of residual blocks per layer
        for _ in range(1, blocks):
            # creates residual blocks and appends to the list 
            layers.append(ResidualBlock(out_channels, out_channels))

        # push out the layered architecture.
        return nn.Sequential(*layers)

    # Connect the architecture into the forward block.
    def forward(self, x):
        """
        Connects the ResNet Architecture 
        Params:
            x (torch.Tensor): Input image tensor of shape [B,in_channels, H, W].
    
        Returns:
            x (torch.Tensor): Segmentation map of shape [B, out_channels, H, W].
        """
        # Encoding (downsampling)
        # Stem
        x = self.stem(x)     #[B, 64, H/4, W/4]

        # residual blocks
        # layer 1 with 2 blocks
        x = self.layer1(x)   #[B, 64, H/4, W/4]
        # layer 2 with 2 blocks
        x = self.layer2(x)   #[B, 128, H/8, W/8]
        # layer 3 with 2 blocks
        x = self.layer3(x)   #[B, 256, H/16, W/16]

        # Decoder (upsampling)
        
        x = self.up1(x)      #[B, 128, H/8, W/8]
        x = self.up2(x)      #[B, 64, H/4, W/4]
        x = self.up3(x)      #[B, 32, H/2, W/2]
        x = self.up4(x)      #[B, 16, H, W]

        return self.out(x)   #[B, out_channels, H, W]
