# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: Reveal.py
@time: 2018/3/20

"""

import torch
import torch.nn as nn

import os

from models.arbedrs import EDRS


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class RevealNet(nn.Module):
    def __init__(self, input_nc, output_nc, nhf=64, norm_layer=nn.InstanceNorm2d, output_function=nn.Sigmoid, cfg=None):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256

        self.cfg = cfg
        cfg.rescale = 'down'
        self.down_net = EDRS(cfg)

        self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
        self.conv5 = nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
        # self.conv6 = nn.Conv2d(nhf, output_nc, 3, 1, 1)
        self.output = output_function()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(True)

        self.linear = nn.Linear(nhf, 30)

        self.norm_layer = norm_layer
        if norm_layer != None:
            self.norm1 = norm_layer(nhf)
            self.norm2 = norm_layer(nhf * 2)
            self.norm3 = norm_layer(nhf * 4)
            self.norm4 = norm_layer(nhf * 2)
            self.norm5 = norm_layer(nhf)

    def forward(self, input, scale):

        lr = self.down_net(input, 1.0 / scale)

        if self.norm_layer != None:
            x = self.relu(self.norm1(self.conv1(lr)))
            x = self.relu(self.norm2(self.conv2(x)))
            x = self.relu(self.norm3(self.conv3(x)))
            x = self.relu(self.norm4(self.conv4(x)))
            x = self.relu(self.norm5(self.conv5(x)))
            x = self.avgpool(x).squeeze_(3).squeeze_(2)
            x = self.output(self.linear(x))
        else:
            x = self.relu(self.conv1(lr))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.relu(self.conv5(x))
            x = self.avgpool(x).squeeze_(3).squeeze_(2)
            x = self.output(self.linear(x))
        return x

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size=30):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = secret_size

        self.decoder = nn.Sequential(
            # # 输入128*128大小的图片
            Conv2D(3, 32, 3, strides=2, activation='relu'),  # 64 * 64
            Conv2D(32, 32, 3, activation='relu'),  # 64
            Conv2D(32, 64, 3, strides=2, activation='relu'),  # 32 * 32
            Conv2D(64, 64, 3, activation='relu'),  # 32
            Conv2D(64, 64, 3, strides=2, activation='relu'),  # 16 * 16
            Conv2D(64, 128, 3, strides=2, activation='relu'),  # 8 * 8
            Conv2D(128, 128, 3, strides=2, activation='relu'),  # 4 * 4
            nn.Flatten())
        # 128*4*4 = 21632
        self.dense1 = Dense(2048, 512, activation='relu')
        self.dense2 = Dense(4608, 512, activation='relu')

        self.tail = Dense(512, secret_size, activation=None)
    def forward(self, image, scale, flag=0):

        # image = self.down_net(image, 1.0 / scale)

        x = self.decoder(image)
        if flag == 1:
            x = self.dense1(x)
        else:
            x = self.dense2(x)
        return torch.sigmoid(self.tail(x))


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            # pytorch默认使用kaiming均匀分布初始化神经网络参数,此处用了正态分布。
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs

if __name__ == "__main__":
    import torch

    # from torchsummary import summary

    Hnet = RevealNet(input_nc=3, output_nc=3)
    inputs = torch.randn(2, 3, 128, 128)
    outputs = Hnet(inputs)
    print(outputs.shape)

    # summary(Hnet, (3, 128, 128))
