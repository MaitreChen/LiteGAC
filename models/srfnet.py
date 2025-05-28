import torch.nn as nn
import torch
from thop import profile


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module that computes spatial attention maps based on average and max pooling.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SRFBasicBlock(nn.Module):
    """
    Basic residual block for SRFNet with spatial attention.
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Initialize the SRFBasicBlock.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
            downsample (nn.Module, optional): Downsample module if needed. Defaults to None.
        """
        super(SRFBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.sa(out) * out

        return out


class SRFNet(nn.Module):
    """
    SRFNet model consisting of initial convolutional layer, residual layers, and classification head.
    """

    def __init__(self, block, blocks_num, num_classes=2):
        """
        Initialize the SRFNet.

        Args:
            block (nn.Module): Basic block class, e.g., SRFBasicBlock.
            blocks_num (list): List of number of blocks in each residual layer.
            num_classes (int, optional): Number of output classes. Defaults to 2.
        """
        super(SRFNet, self).__init__()
        self.in_channel = 32

        # Initial convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.in_channel, kernel_size=7, stride=7),
            nn.GroupNorm(1, self.in_channel, eps=1e-6),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer2 = self._make_layer(block, 64, blocks_num[0])
        self.layer3 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer4 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer5 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.shallow_to_deep_conv = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=3, stride=8, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        low1 = self.layer2(x)
        low2 = self.layer3(low1)
        high1 = self.layer4(low2)
        high2 = self.layer5(high1)

        shallow_to_deep_features = self.shallow_to_deep_conv(x)
        fused = torch.cat((shallow_to_deep_features, high2), dim=1)

        fused = self.avgpool(fused)
        fused = torch.flatten(fused, 1)
        output = self.fc(fused)

        return output


if __name__ == '__main__':
    net = SRFNet(SRFBasicBlock, [1, 1, 3, 1])

    dummy_input = torch.randn(1, 1, 224, 224)
    o = net(dummy_input)
    print(o.shape)

    # Calculate FLOPs and number of parameters
    flops, params = profile(net, inputs=(dummy_input,))
    print("FLOPs:", flops / 1e9, "G")  # Convert FLOPs to GigaFLOPs
    print("Number of parameters:", params / 1e6, "M")  # Convert parameters to MegaParameters
