import torch
import torch.nn as nn
import MinkowskiEngine as ME

class MinkClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MinkClassificationHead, self).__init__()
        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.fc = ME.MinkowskiLinear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc(x)
        return x

class MinkSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MinkSegmentationHead, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels, in_channels // 2, kernel_size=3, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(in_channels // 2)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(in_channels // 2, num_classes, kernel_size=1, dimension=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    # Test MinkClassificationHead
    print("Testing MinkClassificationHead:")
    in_channels = 512
    num_classes = 10
    head = MinkClassificationHead(in_channels, num_classes)
    input_shape = (2, 512, 8, 8, 8)  # Batch size, channels, depth, height, width
    input_tensor = torch.rand(input_shape)
    input_sparse = ME.SparseTensor(input_tensor, coords=torch.nonzero(input_tensor).int())
    output_sparse = head(input_sparse)
    print("Input shape:", input_shape)
    print("Output shape:", output_sparse.F.shape)

    # Test MinkSegmentationHead
    print("\nTesting MinkSegmentationHead:")
    in_channels = 128
    num_classes = 5
    head = MinkSegmentationHead(in_channels, num_classes)
    input_shape = (2, 128, 64, 64, 64)  # Batch size, channels, depth, height, width
    input_tensor = torch.rand(input_shape)
    input_sparse = ME.SparseTensor(input_tensor, coords=torch.nonzero(input_tensor).int())
    output_sparse = head(input_sparse)
    print("Input shape:", input_shape)
    print("Output shape:", output_sparse.F.shape)