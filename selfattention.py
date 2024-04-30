import torch
from torch import nn
import torchvision


class SelfAttention(nn.Module):
  def __init__(self, in_channels):
    super(SelfAttention, self).__init__()
    self.in_channels = in_channels
    self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
    self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
    self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.gamma = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    """
    forward step
    :param x: input tensor
    :return: output tensor
    """
    batch_size, channels, height, width = x.size()

    # Compute query, key, and value
    query = self.query_conv(x).view(batch_size, -1, height * width)
    key = self.key_conv(x).view(batch_size, -1, height * width)
    value = self.value_conv(x).view(batch_size, -1, height * width)

    # Compute attention weights
    attention_weights = nn.functional.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)

    # Apply attention to value
    out = torch.bmm(value, attention_weights.permute(0, 2, 1)).view(batch_size, channels, height, width)

    # Apply scaling factor and residual connection
    out = self.gamma * out + x

    return out
