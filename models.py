
from collections import OrderedDict
from torchvision import models
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

# landmark => 1049 class

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1049)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Dummy(nn.Module):

    def __init__(self):
        super(Dummy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.fc = nn.Linear(64, 1049)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.AdaptiveAvgPool2d(1)(x).squeeze()
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1049).cuda()
    print(model.__repr__())
    exit()
    a=Resnet50().cuda()
    print(a)
    summary(a,(3,224,224))