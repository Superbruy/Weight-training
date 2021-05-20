# -*- coding=utf-8 -*-
'''
# @filename  : shufflev2se.py
# @author    : Superbruy
# @date      : 2021-5-13
# @brief     : shuffle net v2 with se module
'''
import torch
import torch.nn as nn


def conv_bn(input, output, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(input, output):
    return nn.Sequential(
        nn.Conv2d(input, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = int(num_channels / groups)

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit2(nn.Module):
    def __init__(self, input, output, stride, benchmodel):
        super(ShuffleUnit2, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        # split original channels into 2 parts
        oup_inc = output // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(input, input, 3, stride, 1, groups=input, bias=False),
                nn.BatchNorm2d(input),
                # pw-linear
                nn.Conv2d(input, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(input, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

        self.avg = nn.AdaptiveAvgPool2d(1)

    def SEmodule(self, channels, reduction):
        return nn.Sequential(
            nn.Linear(channels, int(channels / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels / reduction), channels, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.branch2(x2))
        elif self.benchmodel == 2:
            out = self._concat(self.branch1(x), self.branch2(x))

        # se module
        b, c, h, w = out.size()
        fc_layer = self.SEmodule(c, 16)
        y = self.avg(out).view(b, c)
        y = fc_layer(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=10, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} multiples is not supported""".format(width_mult))

        # building first layer
        self.input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, self.input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.features = []
        # # building inverted residual blocks
        # for idxstage in range(len(self.stage_repeats)):
        #     numrepeat = self.stage_repeats[idxstage]
        #     output_channel = self.stage_out_channels[idxstage + 2]
        #     for i in range(numrepeat):
        #         if i == 0:
        #             # inp, oup, stride, benchmodel):
        #             self.features.append(ShuffleUnit2(input_channel, output_channel, 2, 2))
        #         else:
        #             self.features.append(ShuffleUnit2(input_channel, output_channel, 1, 1))
        #         input_channel = output_channel

        self.stage2 = self._make_stage(4, self.stage_out_channels[2])
        self.stage3 = self._make_stage(8, self.stage_out_channels[3])
        self.stage4 = self._make_stage(4, self.stage_out_channels[4])

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(self.input_channel, self.stage_out_channels[-1])
        # self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.stage_out_channels[-1], n_class)
        # building classifier
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        x = self.conv_last(x)
        x = self.avg(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.fc(x)
        return x

    def _make_stage(self, num_repeat, out_channels):
        container = []
        for i in range(num_repeat):
            if i == 0:
                container.append(ShuffleUnit2(self.input_channel, out_channels, 2, 2))
            else:
                container.append(ShuffleUnit2(self.input_channel, out_channels, 1, 1))
            self.input_channel = out_channels
        return nn.Sequential(*container)


def shufflenetv2se(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model


if __name__ == "__main__":
    """Testing
    """
    a = torch.rand(10, 3, 28, 28)
    model = ShuffleNetV2()
    b = model(a)
    print(b.shape)
    # print(model.features)
