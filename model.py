
import torch
from torch import nn
import torchvision
from torchvision import transforms, models
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def create_data_loader(batch_size):
  # Split the provided CIFAR-10 train set (50,000 images) into your train and val sets
  # Use the first 40,000 images as your train set and the remaining 10,000 images as val set
  # Use all 10,000 images in the provided test set as your test set
  """
  download CIFAR-10 dataset and create train, validation and test set
  :param batch_size: number of training samples used in one iteration
  :return: train set, test set, validation set
  """

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

  train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

  test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

  train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

  # check data size, should be (C,H,W), class map only useful for visualization and sanity checks
  image_size = train_set[0][0].size()
  class_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
             9: 'truck'}


  return train_loader, val_loader, test_loader

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

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample
    self.attention = SelfAttention(out_channels)          # self attention module

  def forward(self, x):
    """
    forward step
    :param x: input tensor
    :return: output tensor
    """
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out = self.attention(out)

    out += residual
    out = self.relu(out)

    return out

class ResNet(nn.Module):

  def __init__(self, num_block, dropout_prob=0.0, weight_decay=0.0):
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
    self.dropout = nn.Dropout(dropout_prob)
    self.weight_decay = weight_decay

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

def train(train_loader, val_loader, batch_size):
  """
  Train the model using training and validation sets
  :param train_loader: training dataset
  :param val_loader: validation set
  :param batch_size: number of training samples used in one iteration
  """
  model = ResNet([3,4,6,3])
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  # Train the model
  best_accuracy = 0.0
  best_model_path = 'best_model.pth'

  num_epochs = 10
  for epoch in range(num_epochs):
    train_total = 0.0
    train_correct = 0.0
    model.train()

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      train_total += labels.size(0)
      train_correct += (predicted == labels).sum().item()

      # loss every 100 mini-batches
      if i % 100 == 99:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0
    train_accuracy = 100 * train_correct / train_total
    print('Accuracy of the network on the training set: %d %%' % train_accuracy)

    # Validate the model
    model.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    val_accuracy = 0

    with torch.no_grad():
      for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print('Accuracy of the network on the validation set: %d %%' % val_accuracy)

    # Save the best model
    if train_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      torch.save(model.state_dict(), best_model_path)
      print("Best model saved with accuracy: %.2f%%" % best_accuracy)

batch_size = 64
train_loader, val_loader, test_loader = create_data_loader(batch_size)
dropout_prob = 0.5  # Dropout probability
weight_decay = 1e-4  # Weight decay parameter
model = ResNet([3, 4, 6, 3], dropout_prob=dropout_prob, weight_decay=weight_decay)

train(train_loader, val_loader, batch_size)

def test(model_path, test_loader):

  """
  Test the model using testing set
  :param model_path: path to the best model after training
  :param train_loader: test dataset
  """
  dropout_prob = 0.5  # Dropout probability
  weight_decay = 1e-4  # Weight decay parameter
  model = ResNet([3, 4, 6, 3], dropout_prob, weight_decay)

  model.to(device)

  # Load the best model
  model_path = 'best_model.pth'
  model.load_state_dict(torch.load(model_path))
  model.eval()

  # Test the model
  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_loader:
      images, labels = data
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print('Accuracy of the best model on the test images: %.2f %%' % accuracy)

batch_size = 128
train_loader, val_loader, test_loader = create_data_loader(batch_size)

dropout_prob = 0.5  # Dropout probability
weight_decay = 1e-4  # Weight decay parameter
model = ResNet([3, 4, 6, 3], dropout_prob, weight_decay)
model.to(device)

# Load the best model
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

test(model_path, test_loader)

"""1. This time we have implemented self attention in the model. Self attention will try to gather only the relevant information and reducing noise. This can be done by computing attention weights based on the input feature maps and applies them to the value maps.
2. ResidualBlock now has a self attention module added after the 2nd convolution layer.
3. This newly integrated ResNet does perform well while training by achieving accuracy of 48% on training set, 61% on validation set and 57.45% on test images.
4. Given only one cycle of training and validation before testing, the accuracy has increased from our Vanilla Resnet. With longer training period, this model can achieve even higher accuracy.
"""