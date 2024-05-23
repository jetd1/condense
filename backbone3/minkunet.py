import torch
import torch.nn as nn
import MinkowskiEngine as ME

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=stride, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=1, dimension=3)
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, stride=stride, dimension=3),
                ME.MinkowskiBatchNorm(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class MinkUNet34(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MinkUNet34, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 64, kernel_size=7, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True)
        )
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(512, 256, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(256, 128, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(128, 64, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True)
        )
        self.conv5 = ME.MinkowskiConvolution(64, out_channels, kernel_size=1, stride=1, dimension=3)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class MinkUNet14(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MinkUNet14, self).__init__()
        self.inplanes = 32

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 32, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True)
        )
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 128, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 3, stride=2)

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(256, 128, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(128, 64, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True)
        )
        self.conv4 = ME.MinkowskiConvolution(64, out_channels, kernel_size=1, stride=1, dimension=3)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


if __name__ == "__main__":
    # Test MinkUNet34
    print("Testing MinkUNet34:")
    model_34 = MinkUNet34(in_channels=3, out_channels=10)
    input_shape = (2, 3, 64, 64, 64)  # Batch size, channels, depth, height, width
    input_tensor = torch.rand(input_shape)
    input_sparse = ME.SparseTensor(input_tensor, coords=torch.nonzero(input_tensor).int())
    output_sparse = model_34(input_sparse)
    print("Input shape:", input_shape)
    print("Output shape:", output_sparse.F.shape)

    # Test MinkUNet14
    print("\nTesting MinkUNet14:")
    model_14 = MinkUNet14(in_channels=3, out_channels=10)
    input_shape = (2, 3, 64, 64, 64)  # Batch size, channels, depth, height, width
    input_tensor = torch.rand(input_shape)
    input_sparse = ME.SparseTensor(input_tensor, coords=torch.nonzero(input_tensor).int())
    output_sparse = model_14(input_sparse)
    print("Input shape:", input_shape)
    print("Output shape:", output_sparse.F.shape)
