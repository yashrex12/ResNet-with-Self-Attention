import torch
from torch import nn



class ResNet(nn.Module):
  def __init__(self, num_block):
    super(ResNet, self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(64, num_block[0])
    self.layer2 = self._make_layer(128, num_block[1], stride=2)
    self.layer3 = self._make_layer(256, num_block[2], stride=2)
    self.layer4 = self._make_layer(512, num_block[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512, 10)

  def _make_layer(self, out_channels, num_block, stride=1):
    """
    create layers (residual block) of ResNet model
    :param out_channels: output channels for conv layer
    :param num_block: number of residual blocks to be created at each layer
    :param stride: stride for the conv layer (defaults to 1)
    :return: sequence of layers representing the residual blocks
    """
    downsample = None
    if stride != 1 or self.in_channels != out_channels:
      downsample = nn.Sequential(
          nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
      )
    layers = []
    layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels
    for _ in range(1, num_block):
      layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    """
    forward step
    :param x: input tensor
    :return: output tensor
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x
