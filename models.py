import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
import torch.utils as utils


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            # B = x.data.size(0)
            # C = x.data.size(1)
            # H = x.data.size(2)
            # W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super(DownSample1, self).__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        return d5, d4, d3

    def blocks(self):
        return self.down1, self.down2, self.down3, self.down4, self.down5


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = relu(norm(conv(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)

        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = utils.checkpoint.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.relu2(self.norm2(self.conv2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)

        self.norm = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, block_config=(12, 9, 24, 48), compression=0.875,
                 num_init_features=24, bn_size=4, drop_rate=0, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv0 = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
            )

        # Each denseblock
        # self.features = nn.ModuleList()
        # for i, num_layers in enumerate(block_config):
        #     block = _DenseBlock(
        #         num_layers=num_layers,
        #         num_input_features=num_init_features,
        #         bn_size=bn_size,
        #         growth_rate=growth_rate,
        #         drop_rate=drop_rate,
        #         efficient=efficient,
        #     )
        #     self.features.append(block)
        #     num_init_features = num_init_features + num_layers * growth_rate
        #     if i != len(block_config) - 1:
        #         trans = _Transition(
        #             num_input_features=num_init_features,
        #             num_output_features=int(num_init_features * compression)
        #         )
        #         self.features.append(trans)
        #         num_init_features = int(num_init_features * compression)

        self.stage0 = nn.Sequential(
            _DenseBlock(
                num_layers=8,
                num_input_features=num_init_features,
                bn_size=bn_size,
                growth_rate=8,
                drop_rate=drop_rate,
                efficient=efficient,
            ),
            _Transition(
                num_input_features=88,
                num_output_features=64
            )
        )

        self.stage1 = nn.Sequential(
            _DenseBlock(
                num_layers=8,
                num_input_features=64,
                bn_size=bn_size,
                growth_rate=16,
                drop_rate=drop_rate,
                efficient=efficient,
            ),
            _Transition(
                num_input_features=192,
                num_output_features=128
            )
        )

        self.stage2 = nn.Sequential(
            _DenseBlock(
                num_layers=8,
                num_input_features=128,
                bn_size=bn_size,
                growth_rate=32,
                drop_rate=drop_rate,
                efficient=efficient,
            ),
            _Transition(
                num_input_features=384,
                num_output_features=256
            )
        )

        self.stage3 = nn.Sequential(
            _DenseBlock(
                num_layers=8,
                num_input_features=256,
                bn_size=bn_size,
                growth_rate=64,
                drop_rate=drop_rate,
                efficient=efficient,
            ),
            _Transition(
                num_input_features=768,
                num_output_features=512
            )
        )

        self.stage4 = nn.Sequential(
            _DenseBlock(
                num_layers=8,
                num_input_features=512,
                bn_size=bn_size,
                growth_rate=128,
                drop_rate=drop_rate,
                efficient=efficient,
            ),
            _Transition(
                num_input_features=1536,
                num_output_features=1024
            )
        )
        # Final batch norm
        # self.features.append(nn.BatchNorm2d(num_init_features))

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.stage0(x0)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x4, x3, x2


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, anchors, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
            anchor_mask=[0, 1, 2], num_classes=n_classes,
            anchors=anchors.flatten(),
            num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo2 = YoloLayer(
            anchor_mask=[3, 4, 5], num_classes=n_classes,
            anchors=anchors.flatten(),
            num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo3 = YoloLayer(
            anchor_mask=[6, 7, 8], num_classes=n_classes,
            anchors=anchors.flatten(),
            num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])

        else:
            return [x2, x10, x18]


class Yolo_loss(nn.Module):
    def __init__(
            self, image_size=(320, 320), n_classes=80,
            anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
            n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        self.image_size = torch.tensor(image_size)
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = anchors
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = self.image_size // self.strides[i]
            grid_x = torch.arange(fsize[0], dtype=torch.float).repeat(batch, 3, fsize[1], 1).to(device)
            grid_y = torch.arange(fsize[1], dtype=torch.float).repeat(batch, 3, fsize[0], 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize[1], fsize[0], 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize[1], fsize[0], 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize[0], fsize[1], 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize[0], fsize[1]).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize[0], fsize[1], 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize[0], fsize[1], n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = truth_w_all[b, ti]
                    target[b, a, j, i, 3] = truth_h_all[b, ti]
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize[0] / fsize[1])
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2:]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize[0], fsize[1])
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2], weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=pred[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2

    def to(self, *args, **kwargs):
        self.device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        return super(Yolo_loss, self).to(*args, **kwargs)


class Yolov4(nn.Module):
    def __init__(self, backbone, config, inference=False, device=None):
        super(Yolov4, self).__init__()

        output_ch = (4 + 1 + config.classes) * 3

        # backbone
        self.backbone = backbone
        # neck
        self.neek = Neck(inference)
        # yolov4conv137
        if config.pretrained is not None:
            _model = nn.Sequential(*self.backbone.blocks(), self.neek)
            pretrained_dict = torch.load(config.pretrained)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)

        # head
        self.head = Yolov4Head(output_ch, config.classes, config.anchors, inference)
        self.criterion = Yolo_loss(image_size=(config.w, config.h), device=device, batch=config.batch // config.subdivisions, n_classes=config.classes, anchors=config.anchors)

    def forward(self, x, target=None):
        d5, d4, d3 = self.backbone(x)
        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        if target is not None:
            return self.criterion(output, target)
        else:
            return output

    def to(self, *args, **kwargs):
        self.device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        return super(Yolov4, self).to(*args, **kwargs)

    def eval(self):
        self.head.inference = True
        return super(Yolov4, self).eval()


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
            (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = None
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = sys.argv[4]
        width = int(sys.argv[5])
        namesfile = int(sys.argv[6])
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    model = Yolov4(pretrained=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    for i in range(2):  # This 'for' loop is for speed check
        # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    if namesfile == None:
        if n_classes == 20:
            namesfile = 'data/voc.names'
        elif n_classes == 80:
            namesfile = 'data/coco.names'
        else:
            print("please give namefile")

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
