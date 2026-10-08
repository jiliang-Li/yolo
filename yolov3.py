import torch
import torch.nn as nn


__all__ = [
    'Darknet53',
    'Yolov3Neck',
    'Yolov3Head'
]


class Conv(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        """
        自定义一个卷积块，一次性完成卷积+归一化+激活，这在类似于像DarkNet53这样的深层网络编码上可以节省很多代码
        :param c_in: in_channels
        :param c_out: out_channels
        :param k: kernel_size
        :param s:  stride
        :param p: padding
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1),
        )

    def forward(self, entry):
        return self.conv(entry)


class ConvResidual(nn.Module):
    def __init__(self, c_in):       # converlution * 2 + residual
        """
        自定义残差单元，只需给出通道数，该单元完成两次卷积，并进行加残差后返回相同维度的特征图
        :param c_in: 通道数
        """
        c = c_in // 2
        super().__init__()
        # 采用 1*1 + 3*3 的形式加深网络深度，加强特征抽象
        self.conv = nn.Sequential(
            Conv(c_in, c, 1, 1, 0),      # kernel_size = 1进行降通道
            Conv(c, c_in, 3, 1, 1),      # 再用kernel_size = 3把通道升回去
        )

    def forward(self, entry):
        return entry + self.conv(entry)	 # 加残差，既保留原始信息，又融入了提取到的特征


# ---------- ---------- backbone ---------- ----------

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)       # 一个卷积块 = 1层卷积
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3_4 = ConvResidual(64)         # 一个残差块 = 2层卷积
        self.conv5 = Conv(64, 128, 3, 2, 1)
        self.conv6_9 = nn.Sequential(           # = 4层卷积
            ConvResidual(128),
            ConvResidual(128),
        )
        self.conv10 = Conv(128, 256, 3, 2, 1)
        self.conv11_26_2 = nn.Sequential(*[ConvResidual(256) for i in range(8)])    # = 16层卷积
        self.conv27 = Conv(256, 512, 3, 2, 1)
        self.conv28_43_2 = nn.Sequential(*[ConvResidual(512) for i in range(8)])    # = 16层卷积
        self.conv44 = Conv(512, 1024, 3, 2, 1)
        self.conv45_52_2 = nn.Sequential(*[ConvResidual(1024) for i in range(4)])   # = 8层卷积

    def forward(self, entry):
        """
        :param entry:
        :return: c5 c4 c3
        """
        conv1 = self.conv1(entry)
        conv2 = self.conv2(conv1)
        conv3_4 = self.conv3_4(conv2)
        conv5 = self.conv5(conv3_4)
        conv6_9 = self.conv6_9(conv5)
        conv10 = self.conv10(conv6_9)
        conv11_26 = self.conv11_26_2(conv10)
        conv27 = self.conv27(conv11_26)
        conv28_43 = self.conv28_43_2(conv27)
        conv44 = self.conv44(conv28_43)
        conv45_52 = self.conv45_52_2(conv44)
        return conv45_52, conv28_43, conv11_26  # YOLOv3用，所以输出了3次特征


# ---------- ---------- neck ---------- ----------

class NeckConvSet(nn.Module):
    def __init__(self, c_in, c_mid):
        super().__init__()
        self.set = nn.Sequential(
            Conv(c_in, c_mid, 1, 1, 0),
            Conv(c_mid, c_mid * 2, 3, 1, 1),
            Conv(c_mid * 2, c_mid, 1, 1, 0),
            Conv(c_mid, c_mid * 2, 3, 1, 1),
            Conv(c_mid * 2, c_mid, 1, 1, 0),
        )
        self.detect_conv = Conv(c_mid, c_mid * 2, 3, 1, 1)

    def forward(self, entry):
        set_output = self.set(entry)
        return self.detect_conv(set_output), set_output


class NeckUp(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.seq = nn.Sequential(
            Conv(c_in, c_in // 2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, entry):
        return self.seq(entry)


class Yolov3Neck(nn.Module):
    def __init__(self, channel_in=(1024, 512, 256)):
        super().__init__()
        self.conv_set_c5 = NeckConvSet(channel_in[0], channel_in[0] // 2)

        self.up_set_c5 = NeckUp(channel_in[0] // 2)
        self.conv_set_c4 = NeckConvSet(channel_in[0] // 4 + channel_in[1], channel_in[1] // 2)

        self.up_set_c4 = NeckUp(channel_in[1] // 2)
        self.conv_set_c3 = NeckConvSet(channel_in[1] // 4 + channel_in[2], channel_in[2] // 2)

    def forward(self, entry):
        """
        :param entry: c5:((n * 1024 * 13 * 13), c4:(n * 512 * 26 * 26), c3:(n * 256 * 52 * 52))
        :return:
        """
        p5, rout5  = self.conv_set_c5(entry[0])     # p5 = 1024 * 13 * 13, r5 = 512 * 13 * 13
        p5_up = self.up_set_c5(rout5)  # 256 * 26 * 26
        p5up_concat_c4 = torch.cat([p5_up, entry[1]], dim=1)    # 768 * 26 * 26
        p4, rout4 = self.conv_set_c4(p5up_concat_c4)        # p4 = 512 * 26 * 26, r4 = 256 * 26 * 26
        p4_up = self.up_set_c4(rout4)  # 128 * 52 * 52
        p4_concat_c3 = torch.cat([p4_up, entry[2]], dim=1)      # (128 + 256) * 52 * 52
        p3, rout3 = self.conv_set_c3(p4_concat_c3)     #  p3 = 256 * 52 * 52, r3 = 128 * 52 * 52
        return p5, p4, p3


# ---------- ---------- head ---------- ----------

class Yolov3Head(nn.Module):
    def __init__(self, in_channels=(1024, 512, 256), n_classes=80, n_anchors=3):
        super().__init__()
        inc, self.nc, self.na = in_channels, n_classes, n_anchors    # 这行纯为了减短后面3行代码的长度
        self.conv_f5 = nn.Conv2d(inc[0], self.na * (self.nc + 5), 1, 1, 0)
        self.conv_f4 = nn.Conv2d(inc[1], self.na * (self.nc + 5), 1, 1, 0)
        self.conv_f3 = nn.Conv2d(inc[2], self.na * (self.nc + 5), 1, 1, 0)

    def reshape_permute(self, x):
        n, c, h, w = x.shape
        x = x.reshape((n, self.na, self.nc + 5, h, w))
        x = torch.permute(x, (0, 1, 3, 4, 2)).contiguous()  # (n, na, h, w, (nc + 5))
        return x

    def forward(self, entry):
        """
        :param entry: p5, p4, p3
        :return:
        """
        f5 = self.reshape_permute(self.conv_f5(entry[0]))
        f4 = self.reshape_permute(self.conv_f4(entry[1]))
        f3 = self.reshape_permute(self.conv_f3(entry[2]))
        return f5, f4, f3
