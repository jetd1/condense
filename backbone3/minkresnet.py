import torch
import torch.nn as nn
import MinkowskiEngine as ME

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(planes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=3)
        self.bn2 = ME.MinkowskiBatchNorm(planes)
        self.stride = stride

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(inplanes, planes * self.expansion, kernel_size=1, stride=stride, dimension=3),
                ME.MinkowskiBatchNorm(planes * self.expansion)
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(planes)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=3)
        self.bn2 = ME.MinkowskiBatchNorm(planes)
        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1, dimension=3)
        self.bn3 = ME.MinkowskiBatchNorm(planes * self.expansion)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.stride = stride

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(inplanes, planes * self.expansion, kernel_size=1, stride=stride, dimension=3),
                ME.MinkowskiBatchNorm(planes * self.expansion)
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
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class MinkResNet(nn.Module):
    def __init__(self, block, layers, in_channels, out_channels):
        super(MinkResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 64, kernel_size=7, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv5 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(512 * block.expansion, 256 * block.expansion, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(256 * block.expansion),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolutionTranspose(256 * block.expansion, 128 * block.expansion, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128 * block.expansion),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolutionTranspose(128 * block.expansion, 64 * block.expansion, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64 * block.expansion),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(64 * block.expansion, out_channels, kernel_size=1, dimension=3)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        return x

def mink_resnet18(in_channels, out_channels):
    return MinkResNet(BasicBlock, [2, 2, 2, 2], in_channels, out_channels)

def mink_resnet34(in_channels, out_channels):
    return MinkResNet(BasicBlock, [3, 4, 6, 3], in_channels, out_channels)

def mink_resnet50(in_channels, out_channels):
    return MinkResNet(Bottleneck, [3, 4, 6, 3], in_channels, out_channels)

def mink_resnet101(in_channels, out_channels):
    return MinkResNet(Bottleneck, [3, 4, 23, 3], in_channels, out_channels)

if __name__ == "__main__":
    # Test MinkResNet18
    print("Testing MinkResNet18:")
    model = mink_resnet18(in_channels=3, out_channels=10)
    input_shape = (2, 3, 64, 64, 64)  # Batch size, channels, depth, height, width
    input_tensor = torch.rand(input_shape)
    input_sparse = ME.SparseTensor(input_tensor, coords=torch.nonzero(input_tensor).int())
    output_sparse = model(input_sparse)
    print("Input shape:", input_shape)
    print("Output shape:", output_sparse.F.shape)

    # Test MinkResNet50
    print("\nTesting MinkResNet50:")
    model = mink_resnet50(in_channels=3, out_channels=10)
    input_shape = (2, 3, 64, 64, 64)  # Batch size, channels, depth, height, width
    input_tensor = torch.rand(input_shape)
    input_sparse = ME.SparseTensor(input_tensor, coords=torch.nonzero(input_tensor).int())
    output_sparse = model(input_sparse)
    print("Input shape:", input_shape)
    print("Output shape:", output_sparse.F.shape)