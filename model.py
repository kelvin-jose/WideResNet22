from torch import nn as nn


def conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2)


def bn_relu_conv(in_channels, out_channels):
    return nn.Sequential(nn.BatchNorm2d(in_channels),
                         nn.ReLU(inplace=True),
                         conv(in_channels, out_channels))


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.bn_relu = nn.ReLU(nn.BatchNorm2d(input_channels))
        self.conv1 = conv(input_channels, output_channels, stride=stride)
        self.conv2 = bn_relu_conv(output_channels, output_channels)
        self.shortcut = nn.Identity()
        if input_channels != output_channels:
            self.shortcut = conv(input_channels, output_channels, stride=stride, kernel_size=1)

    def forward(self, input):
        x = self.bn_relu(input)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.add_(r)


def construct_group(input_channels, output_channels, n_groups, stride):
    start = [ResidualBlock(input_channels, output_channels, stride)]
    end = [ResidualBlock(output_channels, output_channels) for i in range(1, n_groups)]
    return start + end


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class WideResNet22(nn.Module):
    def __init__(self, n_groups, classes, channel_start=16):
        super().__init__()
        layers = [conv(3, channel_start)]
        n_channels = [channel_start, 96, 192, 384]

        for group in range(n_groups):
            stride = 2 if group > 0 else 1
            layers += construct_group(n_channels[group], n_channels[group + 1], n_groups, stride)

        layers += [nn.BatchNorm2d(n_channels[-1]),
                   nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(1),
                   Flatten(),
                   nn.Linear(n_channels[-1], classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
