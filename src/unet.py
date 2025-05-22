import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A sequence of two convolutional layers each followed by a ReLU activation
    and batch normalization, implementing the common double-convolution block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """
    A downsampling block consisting of a max-pooling operation
    followed by a double-convolution to reduce spatial dimensions
    and increase feature depth.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """
    An upsampling block that first upsamples the feature map (either via
    bilinear interpolation or transpose convolution), pads to match
    dimensions if necessary, concatenates with the corresponding encoder
    feature map, and applies a double-convolution.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Choose upsampling mode: bilinear interpolation vs. transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # No change in channel count for bilinear path
        else:
            # Transposed convolution to learn the upsampling
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: decoder feature map, x2: encoder feature map for skip connection
        x1 = self.up(x1)

        # Compute necessary padding to align spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad symmetrically in height and width dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    A simple 1x1 convolution to project the feature map to the desired
    number of output classes (e.g., one channel for binary segmentation).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(x)

class UNet(nn.Module):
    """
    The U-Net architecture for biomedical image segmentation. It consists of
    an encoding path (contracting) followed by a decoding path (expanding),
    with skip connections between corresponding levels.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder: initial convolution and downsampling stages
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder: upsampling stages with skip connections
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output projection
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Expanding path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Generate segmentation mask
        logits = self.outc(x)
        return torch.sigmoid(logits)  # Apply sigmoid for binary segmentation probabilities
