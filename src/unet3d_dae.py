import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
# FiLM layer
class FiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Linear(2, channels)  # 2 for one-hot T1/T2
        self.beta = nn.Linear(2, channels)

    def forward(self, x, modality):
        gamma = self.gamma(modality)
        beta = self.beta(modality)
        
        return gamma.view(-1, x.size(1), 1, 1, 1) * x + beta.view(-1, x.size(1), 1, 1, 1) # Reshape for 3D
    
# Weight-Standardized 3D Convolution
class WSConv3d(nn.Conv3d):
    def forward(self, x):
        weight = self.weight

        # Compute mean and std per output channel
        mean = weight.mean(dim=(1,2,3,4), keepdim=True)
        std = weight.std(dim=(1,2,3,4), keepdim=True) + 1e-5  # Add epsilon for numerical stability

        # Normalize weights
        weight = (weight - mean) / std

        # Perform convolution with standardized weights
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
# Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = WSConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.film1 = FiLM(out_channels) 
        self.swish1 = Swish()

        self.conv2 = WSConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.film2 = FiLM(out_channels) 
        self.swish2 = Swish()

    def forward(self, x, modality):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, modality)
        x = self.swish1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, modality)
        x = self.swish2(x)
        
        return x
    
# Upsampling Block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # Concatenation doubles channels

    def forward(self, x, skip, modality):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.conv(x, modality)
        return x
    
# Downsampling Block 
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x, modality):
        x = self.conv(x, modality)
        p = self.pool(x)
        return x, p 

# Full 3D U-Net
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        self.down1 = DownBlock(in_channels, base_filters)
        self.down2 = DownBlock(base_filters, base_filters * 2)
        self.down3 = DownBlock(base_filters * 2, base_filters * 4)

        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

        self.up3 = UpBlock(base_filters * 8, base_filters * 4)
        self.up2 = UpBlock(base_filters * 4, base_filters * 2)
        self.up1 = UpBlock(base_filters * 2, base_filters)

        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)  # 1x1x1 to get output

    def forward(self, x, modality):
        s1, p1 = self.down1(x, modality)
        s2, p2 = self.down2(p1, modality)
        s3, p3 = self.down3(p2, modality)

        b = self.bottleneck(p3, modality)

        u3 = self.up3(b, s3, modality)
        u2 = self.up2(u3, s2, modality)
        u1 = self.up1(u2, s1, modality)

        out = self.final_conv(u1)
        return out 