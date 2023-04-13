import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.conv import Conv2d
import nets.resnet50


class Renshen(nn.Module):
    def __init__(self):
        super(Renshen, self).__init__()
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


dataset = torchvision.datasets.CIFAR10("./dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)
renshen = Renshen()
print(renshen)
writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = renshen(imgs)
    print(imgs.shape)
    # print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
