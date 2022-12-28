# Code taken from https://github.com/meliketoy/wide-resnet.pytorch
# with some modifications
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable

from my_utils.pytorch1_utils.initialization import kaiming_normal_, kaiming_uniform_

WEIGHT_INIT = kaiming_uniform_
BIAS_INIT = partial(init.constant_, val=0)


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "The depth of a ResNet must be " \
        "one of {}. Found {}!".format(depth_lst, depth)

    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


def conv3x3(in_channels, out_channels, stride=1,
            weight_initializer=WEIGHT_INIT,
            bias=False, bias_initializer=BIAS_INIT):
    # Note that padding=1 (same padding) here!
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=bias)
    weight_initializer(conv.weight)
    if bias:
        bias_initializer(conv.bias)
    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 weight_initializer=WEIGHT_INIT,
                 bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):

        super(BasicBlock, self).__init__()

        # (c_in, h, w) => (c_out, h/s, w/s)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride,
                             weight_initializer=weight_initializer,
                             bias=bias, bias_initializer=bias_initializer)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        # (c_out, h/s, w/s) => (c_out, h/s, w/s)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1,
                             weight_initializer=weight_initializer,
                             bias=bias, bias_initializer=bias_initializer)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        if stride != 1 or in_channels != self.expansion * out_channels:
            # (c_in, h, w) => (c_out, h/s, w/s)
            conv3 = nn.Conv2d(in_channels, self.expansion * out_channels,
                              kernel_size=1, stride=stride, bias=bias)
            weight_initializer(conv3.weight)
            if bias:
                bias_initializer(conv3.bias)

            bn3 = nn.BatchNorm2d(self.expansion * out_channels, momentum=bn_momentum)

            self.shortcut = nn.Sequential(OrderedDict([
                ('conv3', conv3), ('bn3', bn3)]))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1,
                 weight_initializer=WEIGHT_INIT,
                 bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):

        super(Bottleneck, self).__init__()

        # (c_in, h, w) => (c_out, h, w)
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        weight_initializer(conv1.weight)
        if bias:
            bias_initializer(conv1.bias)
        self.conv1 = conv1
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        # Conv2
        # --------------------- #
        # (c_out, h, w) => (c_out, h/s, w/s)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride,
                             weight_initializer=weight_initializer,
                             bias=bias, bias_initializer=bias_initializer)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        # --------------------- #

        # Conv3
        # --------------------- #
        # (c_out, h/s, w/s) => (c_out * expansion, h/s, w/s)
        conv3 = nn.Conv2d(out_channels, self.expansion * out_channels,
                          kernel_size=1, bias=bias)
        weight_initializer(conv3.weight)
        if bias:
            bias_initializer(conv3.bias)
        self.conv3 = conv3
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels, momentum=bn_momentum)
        # --------------------- #

        if stride != 1 or in_channels != self.expansion * out_channels:
            # (c_in, h, w) => (c_out, h/s, w/s)
            conv4 = nn.Conv2d(in_channels, self.expansion * out_channels,
                              kernel_size=1, stride=stride, bias=bias)
            weight_initializer(conv4.weight)
            if bias:
                bias_initializer(conv4.bias)

            bn4 = nn.BatchNorm2d(self.expansion * out_channels, momentum=bn_momentum)

            self.shortcut = nn.Sequential(
                OrderedDict([('conv4', conv4), ('bn4', bn4)]))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, feat_dim=0, in_channels=3, base_out_channels=64,
                 weight_initializer=WEIGHT_INIT, bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):
        super(ResNet, self).__init__()

        ic = in_channels
        boc = base_out_channels
        block, num_blocks = cfg(depth)

        self._in_planes = boc

        # (3, 32, 32) => (64, 32, 32)
        self.conv1 = conv3x3(ic, boc, stride=1, weight_initializer=weight_initializer,
                             bias=bias, bias_initializer=bias_initializer)
        self.bn1 = nn.BatchNorm2d(boc, momentum=bn_momentum)

        # (64, 32, 32) => (64*exp, 32, 32)
        self.layer1 = self._make_layer(block, boc, num_blocks[0], stride=1,
                                       weight_initializer=weight_initializer,
                                       bias=bias, bias_initializer=bias_initializer,
                                       bn_momentum=bn_momentum)
        # (64*exp, 32, 32) => (128*exp, 16, 16)
        self.layer2 = self._make_layer(block, 2*boc, num_blocks[1], stride=2,
                                       weight_initializer=weight_initializer,
                                       bias=bias, bias_initializer=bias_initializer,
                                       bn_momentum=bn_momentum)
        # (128*exp, 16, 16) => (256*exp, 8, 8)
        self.layer3 = self._make_layer(block, 4*boc, num_blocks[2], stride=2,
                                       weight_initializer=weight_initializer,
                                       bias=bias, bias_initializer=bias_initializer,
                                       bn_momentum=bn_momentum)
        # (256*exp, 16, 16) => (512*exp, 4, 4)
        self.layer4 = self._make_layer(block, 8*boc, num_blocks[3], stride=2,
                                       weight_initializer=weight_initializer,
                                       bias=bias, bias_initializer=bias_initializer,
                                       bn_momentum=bn_momentum)

        self.last_dim = 8 * boc * block.expansion
        self.feat_dim = feat_dim

        if self.feat_dim > 0:
            linear = nn.Linear(self.last_dim, self.feat_dim, bias=True)
            weight_initializer(linear.weight)
            bias_initializer(linear.bias)
            self.linear = linear
        else:
            self.linear = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride,
                    weight_initializer, bias, bias_initializer, bn_momentum):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self._in_planes, planes, stride,
                                weight_initializer=weight_initializer,
                                bias=bias, bias_initializer=bias_initializer,
                                bn_momentum=bn_momentum))
            self._in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 4, f"x.size()={x.size()}!"
        b, c, h, w = x.size()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        out = self.layer2(out)
        h = h//2; w = w//2

        out = self.layer3(out)
        h = h//2; w = w//2

        out = self.layer4(out)
        h = h//2; w = w//2

        out = F.avg_pool2d(out, (h, w))
        out = out.view(b, self.last_dim)
        out = self.linear(out)

        return out

    def get_last_layer(self):
        return self.linear


class ResNet18(ResNet):
    def __init__(self, feat_dim=0, in_channels=3, base_out_channels=64,
                 weight_initializer=WEIGHT_INIT, bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):
        super(ResNet18, self).__init__(
            18, feat_dim, in_channels, base_out_channels,
            weight_initializer=weight_initializer,
            bias=bias, bias_initializer=bias_initializer,
            bn_momentum=bn_momentum)


class ResNet34(ResNet):
    def __init__(self, feat_dim=0, in_channels=3, base_out_channels=64,
                 weight_initializer=WEIGHT_INIT, bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):
        super(ResNet34, self).__init__(
            34, feat_dim, in_channels, base_out_channels,
            weight_initializer=weight_initializer,
            bias=bias, bias_initializer=bias_initializer,
            bn_momentum=bn_momentum)


class ResNet50(ResNet):
    def __init__(self, feat_dim=0, in_channels=3, base_out_channels=64,
                 weight_initializer=WEIGHT_INIT, bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):
        super(ResNet50, self).__init__(
            50, feat_dim, in_channels, base_out_channels,
            weight_initializer=weight_initializer,
            bias=bias, bias_initializer=bias_initializer,
            bn_momentum=bn_momentum)


class ResNet101(ResNet):
    def __init__(self, feat_dim=0, in_channels=3, base_out_channels=64,
                 weight_initializer=WEIGHT_INIT, bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):
        super(ResNet101, self).__init__(
            101, feat_dim, in_channels, base_out_channels,
            weight_initializer=weight_initializer,
            bias=bias, bias_initializer=bias_initializer,
            bn_momentum=bn_momentum)


class ResNet152(ResNet):
    def __init__(self, feat_dim=0, in_channels=3, base_out_channels=64,
                 weight_initializer=WEIGHT_INIT, bias=False, bias_initializer=BIAS_INIT,
                 bn_momentum=0.1):
        super(ResNet152, self).__init__(
            152, feat_dim, in_channels, base_out_channels,
            weight_initializer=weight_initializer,
            bias=bias, bias_initializer=bias_initializer,
            bn_momentum=bn_momentum)


if __name__ == '__main__':
    net = ResNet(50, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())
