import torch
import torch.nn as nn
from torch.nn import functional as F

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
    def __init__(self, c_in):  # converlution * 2 + residual
        """
        自定义残差单元，只需给出通道数，该单元完成两次卷积，并进行加残差后返回相同维度的特征图
        :param c_in: 通道数
        """
        c = c_in // 2
        super().__init__()
        # 采用 1*1 + 3*3 的形式加深网络深度，加强特征抽象
        self.conv = nn.Sequential(
            Conv(c_in, c, 1, 1, 0),  # kernel_size = 1进行降通道
            Conv(c, c_in, 3, 1, 1),  # 再用kernel_size = 3把通道升回去
        )

    def forward(self, entry):
        return entry + self.conv(entry)  # 加残差，既保留原始信息，又融入了提取到的特征


# ---------- ---------- backbone ---------- ----------

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)  # 一个卷积块 = 1层卷积
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3_4 = ConvResidual(64)  # 一个残差块 = 2层卷积
        self.conv5 = Conv(64, 128, 3, 2, 1)
        self.conv6_9 = nn.Sequential(  # = 4层卷积
            ConvResidual(128),
            ConvResidual(128),
        )
        self.conv10 = Conv(128, 256, 3, 2, 1)
        self.conv11_26_2 = nn.Sequential(*[ConvResidual(256) for i in range(8)])  # = 16层卷积
        self.conv27 = Conv(256, 512, 3, 2, 1)
        self.conv28_43_2 = nn.Sequential(*[ConvResidual(512) for i in range(8)])  # = 16层卷积
        self.conv44 = Conv(512, 1024, 3, 2, 1)
        self.conv45_52_2 = nn.Sequential(*[ConvResidual(1024) for i in range(4)])  # = 8层卷积

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
        p5, rout5 = self.conv_set_c5(entry[0])  # p5 = 1024 * 13 * 13, r5 = 512 * 13 * 13
        p5_up = self.up_set_c5(rout5)  # 256 * 26 * 26
        p5up_concat_c4 = torch.cat([p5_up, entry[1]], dim=1)  # 768 * 26 * 26
        p4, rout4 = self.conv_set_c4(p5up_concat_c4)  # p4 = 512 * 26 * 26, r4 = 256 * 26 * 26
        p4_up = self.up_set_c4(rout4)  # 128 * 52 * 52
        p4_concat_c3 = torch.cat([p4_up, entry[2]], dim=1)  # (128 + 256) * 52 * 52
        p3, rout3 = self.conv_set_c3(p4_concat_c3)  # p3 = 256 * 52 * 52, r3 = 128 * 52 * 52
        return p5, p4, p3


# ---------- ---------- head ---------- ----------

class Yolov3Head(nn.Module):
    def __init__(self, in_channels=(1024, 512, 256), n_classes=80, n_anchors=3):
        super().__init__()
        inc, self.nc, self.na = in_channels, n_classes, n_anchors  # 这行纯为了减短后面3行代码的长度
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


# ---------- ---------- Layer、Loss ---------- ----------

class Yolov3Layer(nn.Module):
    """
    YOLOv3 [yolo] layer 的 PyTorch 实现。

    对应原始 Darknet YOLOv3 中 yolo_layer.c 的核心行为：

    1. forward():
        对 Head 的 raw prediction 做原版 activation：

        tx  -> sigmoid
        ty  -> sigmoid
        tw  -> raw
        th  -> raw
        obj -> sigmoid
        cls -> sigmoid

    2. decode():
        将已经 activation 后的预测解码成 bbox：

        bx = (tx + grid_x) * stride
        by = (ty + grid_y) * stride
        bw = exp(tw) * anchor_w
        bh = exp(th) * anchor_h

    输入顺序固定为：

        f5 -> stride 32 -> 13x13 -> mask (6,7,8)
        f4 -> stride 16 -> 26x26 -> mask (3,4,5)
        f3 -> stride  8 -> 52x52 -> mask (0,1,2)

    输入 tensor:
        [B, 3, H, W, nc+5]

    最后一维:
        [tx, ty, tw, th, obj, cls0, cls1, ...]
    """

    def __init__(
            self,
            n_classes=80,
            anchors=(
                    (10, 13),
                    (16, 30),
                    (33, 23),

                    (30, 61),
                    (62, 45),
                    (59, 119),

                    (116, 90),
                    (156, 198),
                    (373, 326),
            ),
            masks=(
                    (6, 7, 8),  # f5, stride=32
                    (3, 4, 5),  # f4, stride=16
                    (0, 1, 2),  # f3, stride=8
            ),
            strides=(32, 16, 8),
    ):
        super().__init__()

        self.nc = n_classes
        self.no = self.nc + 5

        self.masks = tuple(tuple(x) for x in masks)
        self.strides = tuple(strides)

        assert len(self.masks) == 3
        assert len(self.strides) == 3

        self.na = len(self.masks[0])

        for mask in self.masks:
            assert len(mask) == self.na

        anchors = torch.tensor(anchors, dtype=torch.float32)

        # anchor不是可训练参数，
        # 但需要随着model自动进行CPU/CUDA迁移
        self.register_buffer("anchors", anchors)

    # ------------------------------------------------------------
    # grid
    # ------------------------------------------------------------

    @staticmethod
    def make_grid(h, w, device, dtype):
        """
        创建grid。

        返回:
            [1, 1, H, W, 2]

        最后一维:
            [grid_x, grid_y]

        例如H=W=3：

        grid_x:
            0 1 2
            0 1 2
            0 1 2

        grid_y:
            0 0 0
            1 1 1
            2 2 2
        """
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                        torch.arange(w, device=device, dtype=dtype), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.reshape(1, 1, h, w, 2)

    # ------------------------------------------------------------
    # activation
    # ------------------------------------------------------------

    def activate_one(self, x):
        """
        对一个尺度执行原始 YOLOv3 activation。

        输入:
            [B, 3, H, W, nc+5]

        raw:
            tx, ty, tw, th, obj, cls...

        输出:
            sigmoid(tx)
            sigmoid(ty)
            tw
            th
            sigmoid(obj)
            sigmoid(cls...)
        """

        assert x.ndim == 5
        assert x.shape[1] == self.na
        assert x.shape[-1] == self.no

        y = x.clone()
        # x、y中心偏移
        y[..., 0:2] = torch.sigmoid(x[..., 0:2])
        # w、h保持raw
        # y[..., 2:4] = x[..., 2:4]
        # objectness + classes
        y[..., 4:] = torch.sigmoid(x[..., 4:])
        return y

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------

    def forward(self, entry):
        """
        模拟原始 Darknet forward_yolo_layer
        中 prediction activation 的部分。

        entry:
            (
                f5,     # stride32
                f4,     # stride16
                f3,     # stride8
            )

        注意：
            entry必须是Head输出的RAW prediction。

        返回:
            (
                f5_act,
                f4_act,
                f3_act,
            )

        其中：

            xy  已sigmoid
            wh  保持raw
            obj 已sigmoid
            cls 已sigmoid

        forward()本身不做bbox坐标decode。
        """

        assert len(entry) == 3
        outputs = []
        for x in entry:
            outputs.append(self.activate_one(x))
        return tuple(outputs)

    # ------------------------------------------------------------
    # decode one level
    # ------------------------------------------------------------

    def decode_one(self, x, level, normalized=False):
        """
        解码一个尺度。

        重要：
            x必须已经经过forward()/activate_one()。

            这里不会再次sigmoid。

        参数
        ----------
        x:
            activated prediction:

            [B, 3, H, W, nc+5]

            最后一维：
                [
                    sigmoid(tx),
                    sigmoid(ty),
                    tw,
                    th,
                    sigmoid(obj),
                    sigmoid(cls...)
                ]

        level:
            0 -> f5 -> stride32 -> mask(6,7,8)
            1 -> f4 -> stride16 -> mask(3,4,5)
            2 -> f3 -> stride8  -> mask(0,1,2)

        normalized:
            False:
                bbox为网络输入图像上的pixel坐标：

                [cx_pixel,
                 cy_pixel,
                 w_pixel,
                 h_pixel]

            True:
                bbox归一化到网络输入宽高：

                [cx_norm,
                 cy_norm,
                 w_norm,
                 h_norm]

        返回
        ----------
        [B, 3, H, W, nc+5]

        最后一维：
            [cx, cy, w, h, obj, cls...]
        """

        assert 0 <= level < 3

        assert x.ndim == 5
        assert x.shape[1] == self.na
        assert x.shape[-1] == self.no

        _, na, h, w, _ = x.shape

        # ------------------------------------------------
        # 1. grid
        # ------------------------------------------------

        grid = self.make_grid(h=h, w=w, device=x.device, dtype=x.dtype)

        # ------------------------------------------------
        # 2. 当前level使用的anchor
        #
        # level0 -> [6,7,8]
        # level1 -> [3,4,5]
        # level2 -> [0,1,2]
        # ------------------------------------------------

        mask = self.masks[level]
        anchor = self.anchors[list(mask)]
        anchor = anchor.to(dtype=x.dtype)
        anchor = anchor.reshape(1, na, 1, 1, 2)
        stride = self.strides[level]

        # ------------------------------------------------
        # 3. decode xy
        #
        # x[...,0:2]已经经过sigmoid
        #
        # bx = sigmoid(tx) + grid_x
        # by = sigmoid(ty) + grid_y
        #
        # 转换成网络输入pixel：
        #
        # bx *= stride
        # by *= stride
        # ------------------------------------------------

        xy = (x[..., 0:2] + grid) * stride

        # ------------------------------------------------
        # 4. decode wh
        #
        # bw = exp(tw) * anchor_w
        # bh = exp(th) * anchor_h
        # ------------------------------------------------

        wh = torch.exp(x[..., 2:4]) * anchor

        # ------------------------------------------------
        # 5. normalized bbox
        # ------------------------------------------------

        if normalized:
            net_w = w * stride
            net_h = h * stride
            scale = torch.tensor([net_w, net_h], device=x.device, dtype=x.dtype).reshape(1, 1, 1, 1, 2)
            xy = xy / scale
            wh = wh / scale

        # ------------------------------------------------
        # 6. bbox + obj + cls
        # ------------------------------------------------
        output = torch.cat((xy, wh, x[..., 4:]), dim=-1)
        return output

    # ------------------------------------------------------------
    # decode all levels
    # ------------------------------------------------------------

    def decode(self, entry, normalized=False, flatten=False):
        """
        解码三个尺度。

        重要：
            entry必须已经经过forward()。

        也就是：

            raw = head(...)
            activated = yolo_layer(raw)
            decoded = yolo_layer.decode(activated)

        entry:
            (
                f5_act,     # 13x13
                f4_act,     # 26x26
                f3_act,     # 52x52
            )

        flatten=False:

            返回：

            (
                f5_decoded,
                f4_decoded,
                f3_decoded,
            )

        flatten=True:

            返回：

                [B, N, nc+5]

        416x416时：

            N =
                3*13*13
              + 3*26*26
              + 3*52*52

              = 10647
        """

        assert len(entry) == 3

        outputs = []

        for level, x in enumerate(entry):
            y = self.decode_one(x, level=level, normalized=normalized)
            outputs.append(y)

        if not flatten:
            return tuple(outputs)

        batch_size = outputs[0].shape[0]
        outputs = [x.reshape(batch_size, -1, self.no) for x in outputs]
        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------
    # inference convenience API
    # ------------------------------------------------------------

    def inference(self, entry, normalized=False, flatten=True):
        """
        推理快捷接口。

        entry:
            Head输出的raw prediction。

        等价于：

            activated = self.forward(entry)

            decoded = self.decode(
                activated
            )

        返回decode后的bbox。
        """

        activated = self.forward(entry)
        return self.decode(activated, normalized=normalized, flatten=flatten)


class Yolov3Loss(nn.Module):
    """
    尽量按照原始 Darknet YOLOv3 yolo_layer.c 的训练逻辑实现。

    ------------------------------------------------------------
    输入
    ------------------------------------------------------------

    entry:
        Head 的 RAW 输出，注意不能先经过 Yolov3Layer.forward():

        (
            f5,     # [B, 3, H/32, W/32, nc+5]
            f4,     # [B, 3, H/16, W/16, nc+5]
            f3,     # [B, 3, H/8,  W/8,  nc+5]
        )

        最后一维:
            [tx, ty, tw, th, obj, cls0, cls1, ...]

        全部都是 raw logits / raw values。

    targets:
        原始 GT，归一化到输入图像宽高。

        单张图:
            [N, 5]

            [
                cls,
                cx,
                cy,
                w,
                h
            ]

        batch:
            [N, 6]

            [
                batch_idx,
                cls,
                cx,
                cy,
                w,
                h
            ]

    positive_targets:
        可选。

        如果传入，可以直接使用你现在生成的 all_data2：

        单图:
            [N, 8]

            [
                cls,
                anchor_idx,
                grid_x,
                grid_y,
                offset_x,
                offset_y,
                offset_w,
                offset_h
            ]

        batch:
            [N, 9]

            [
                batch_idx,
                cls,
                anchor_idx,
                grid_x,
                grid_y,
                offset_x,
                offset_y,
                offset_w,
                offset_h
            ]

        如果 positive_targets=None，
        本类会直接从 targets 重新进行原版 best-anchor assignment。

    ------------------------------------------------------------
    原版YOLOv3 loss/delta逻辑
    ------------------------------------------------------------

    positive:

        xy:
            target = GT在grid cell内部的偏移
            使用 BCEWithLogits

            这样对raw logit求梯度：
                sigmoid(raw) - target

            与Darknet：
                delta = target - sigmoid(raw)

            在梯度方向上等价。

        wh:
            target:
                tw = log(gt_w / anchor_w)
                th = log(gt_h / anchor_h)

            使用 0.5 * MSE

        bbox整体scale:
            2 - gt_w * gt_h

        obj:
            positive target = 1

        cls:
            one-hot BCEWithLogits

    negative:

        obj target = 0

    ignore:

        如果当前预测bbox与任意GT：

            IoU > ignore_thresh

        则不计算negative objectness loss。

    ------------------------------------------------------------
    注意
    ------------------------------------------------------------

    Darknet中打印的cost实际上是sum(delta^2)，
    但backward直接传播delta。

    因此这里构造的是一个PyTorch surrogate loss，
    目标是让autograd产生与Darknet delta一致的梯度，
    而不是机械复现Darknet打印出来的cost数值。
    """

    def __init__(
        self,
        n_classes=80,
        anchors=(
            (10, 13),
            (16, 30),
            (33, 23),

            (30, 61),
            (62, 45),
            (59, 119),

            (116, 90),
            (156, 198),
            (373, 326),
        ),
        masks=(
            (6, 7, 8),     # f5
            (3, 4, 5),     # f4
            (0, 1, 2),     # f3
        ),
        strides=(32, 16, 8),

        ignore_thresh=0.7,
        truth_thresh=1.0,
    ):
        super().__init__()

        self.nc = n_classes
        self.no = self.nc + 5

        self.masks = tuple(tuple(x) for x in masks)

        self.strides = tuple(strides)

        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh

        anchors = torch.tensor(anchors, dtype=torch.float32)

        self.register_buffer("anchors", anchors)

        assert len(self.masks) == 3
        assert len(self.strides) == 3

        for mask in self.masks:
            assert len(mask) == 3

    # ============================================================
    # basic utils
    # ============================================================

    @staticmethod
    def make_grid(h, w, device, dtype):

        gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                torch.arange(w, device=device, dtype=dtype), indexing="ij")
        grid = torch.stack((gx, gy), dim=-1)
        return grid.reshape(1, 1, h, w, 2)

    @staticmethod
    def xywh_to_xyxy(box):
        """
        box:
            [..., 4]

            cx, cy, w, h

        return:
            x1, y1, x2, y2
        """

        xy = box[..., 0:2]
        wh = box[..., 2:4]
        half = wh / 2
        return torch.cat((xy - half, xy + half), dim=-1)

    @staticmethod
    def box_iou(box1, box2):
        """
        box1:
            [N, 4], xywh normalized

        box2:
            [M, 4], xywh normalized

        return:
            [N, M]
        """

        if box1.numel() == 0 or box2.numel() == 0:
            return box1.new_zeros((box1.shape[0], box2.shape[0]))

        b1 = Yolov3Loss.xywh_to_xyxy(box1)
        b2 = Yolov3Loss.xywh_to_xyxy(box2)
        lt = torch.maximum(b1[:, None, 0:2], b2[None, :, 0:2])
        rb = torch.minimum(b1[:, None, 2:4], b2[None, :, 2:4])
        wh = (rb - lt).clamp(min=0)
        inter = (wh[..., 0] * wh[..., 1])
        area1 = ((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))
        area2 = ((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))
        union = (area1[:, None] + area2[None, :] - inter)
        return inter / (union + 1e-16)

    @staticmethod
    def wh_iou(wh1, wh2):
        """
        wh1:
            [N, 2]

        wh2:
            [M, 2]

        两者中心视为相同，只比较WH。
        """

        wh1 = wh1[:, None, :]
        wh2 = wh2[None, :, :]

        inter = torch.minimum(wh1, wh2).prod(dim=2)

        area1 = wh1.prod(dim=2)
        area2 = wh2.prod(dim=2)

        return inter / (area1 + area2 - inter + 1e-16)

    # ============================================================
    # target normalization
    # ============================================================

    def prepare_targets(self, targets, device, dtype):
        """
        统一成：

            [N, 6]

            batch_idx,
            cls,
            cx,
            cy,
            w,
            h
        """

        if targets is None:
            return torch.zeros((0, 6), device=device, dtype=dtype)

        targets = torch.as_tensor(targets, device=device, dtype=dtype)

        if targets.numel() == 0:
            return targets.reshape(0, 6)

        if targets.ndim != 2:
            raise ValueError("targets必须为[N,5]或[N,6]")

        # 单张图片
        if targets.shape[1] == 5:
            batch_idx = torch.zeros((targets.shape[0], 1), device=device, dtype=dtype)
            targets = torch.cat((batch_idx, targets), dim=1)
        elif targets.shape[1] != 6:
            raise ValueError("targets格式应为 [cls,cx,cy,w,h] 或 [batch,cls,cx,cy,w,h]")

        return targets

    def prepare_positive_targets(self, positive_targets, device, dtype):
        """
        统一成：

            [N, 9]

            batch,
            cls,
            anchor_idx,
            grid_x,
            grid_y,
            tx,
            ty,
            tw,
            th
        """

        positive_targets = torch.as_tensor(positive_targets, device=device, dtype=dtype)
        if positive_targets.numel() == 0:
            return positive_targets.reshape(0, 9)

        # 用户当前all_data2
        if positive_targets.shape[1] == 8:
            batch_idx = torch.zeros((positive_targets.shape[0], 1), device=device, dtype=dtype)
            positive_targets = torch.cat((batch_idx, positive_targets), dim=1)
        elif positive_targets.shape[1] != 9:
            raise ValueError("positive_targets必须为[N,8]或[N,9]")

        return positive_targets

    # ============================================================
    # anchor mapping
    # ============================================================

    def anchor_to_level(self, anchor_idx):
        """
        global anchor idx
            ->
        level + local anchor idx
        """

        for level, mask in enumerate(self.masks):
            if anchor_idx in mask:
                return (level, mask.index(anchor_idx))

        raise ValueError(f"anchor {anchor_idx} 不存在于masks中")

    # ============================================================
    # decode current prediction
    # ============================================================

    def decode_level_normalized(self, raw, level):
        """
        将一个level的raw prediction decode成
        normalized:

            [cx, cy, w, h]

        用于当前prediction与所有GT算IoU，
        决定negative / ignore。

        raw:
            [B,3,H,W,nc+5]

        return:
            [B,3,H,W,4]
        """

        B, A, H, W, _ = raw.shape
        grid = self.make_grid(H, W, raw.device, raw.dtype)
        mask = self.masks[level]
        anchor = self.anchors[list(mask)].to(device=raw.device, dtype=raw.dtype)
        anchor = anchor.reshape(1, A, 1, 1, 2)
        stride = self.strides[level]
        net_w = W * stride
        net_h = H * stride

        # ------------------------------------------------
        # xy normalized
        # ------------------------------------------------

        xy = (torch.sigmoid(raw[..., 0:2]) + grid)
        grid_scale = torch.tensor([W, H], device=raw.device, dtype=raw.dtype).reshape(1, 1, 1, 1, 2)
        xy = xy / grid_scale

        # ------------------------------------------------
        # wh normalized
        # ------------------------------------------------
        image_scale = torch.tensor([net_w, net_h], device=raw.device, dtype=raw.dtype).reshape(1, 1, 1, 1, 2)
        wh = (torch.exp(raw[..., 2:4]) * anchor / image_scale)
        return torch.cat((xy, wh), dim=-1)

    # ============================================================
    # allocate dense training tensors
    # ============================================================

    def make_level_targets(self, entry):
        """
        每个level创建：

            positive mask
            box target
            class target
            box scale
            ignore mask
        """

        result = []
        for raw in entry:
            B, A, H, W, _ = raw.shape
            positive = torch.zeros((B, A, H, W), device=raw.device, dtype=torch.bool)
            box_target = torch.zeros((B, A, H, W, 4), device=raw.device, dtype=raw.dtype)
            cls_target = torch.zeros((B, A, H, W, self.nc), device=raw.device, dtype=raw.dtype)
            box_scale = torch.zeros((B, A, H, W), device=raw.device, dtype=raw.dtype)
            ignore = torch.zeros((B, A, H, W), device=raw.device, dtype=torch.bool)
            result.append({
                "positive": positive,
                "box_target": box_target,
                "cls_target": cls_target,
                "box_scale": box_scale,
                "ignore": ignore,
            })
        return result

    # ============================================================
    # ignore / truth_thresh
    # ============================================================

    def build_ignore_and_truth_thresh(self, entry, targets, level_data):
        """
        对所有prediction：

            decode prediction
                ↓
            与当前图片所有GT算IoU
                ↓
            best_iou

        best_iou > ignore_thresh:
            ignore negative obj loss

        best_iou > truth_thresh:
            原版还会把这个prediction作为positive。

        默认YOLOv3:
            truth_thresh = 1

        因此正常情况下不会触发第二个分支。
        """

        for level, raw in enumerate(entry):
            B, A, H, W, _ = raw.shape
            pred_boxes = (self.decode_level_normalized(raw, level))

            for b in range(B):
                gt_b = targets[targets[:, 0].long() == b]
                if gt_b.shape[0] == 0:
                    continue

                gt_boxes = gt_b[:, 2:6]
                pred_flat = pred_boxes[b].reshape(-1, 4)
                ious = self.box_iou(pred_flat, gt_boxes)
                best_iou, best_gt_idx = (ious.max(dim=1))
                best_iou = best_iou.reshape(A, H, W)
                best_gt_idx = best_gt_idx.reshape(A, H, W)

                # ----------------------------------------
                # ignore negative objectness
                # ----------------------------------------
                level_data[level]["ignore"][b] = (best_iou > self.ignore_thresh)

                # ----------------------------------------
                # truth_thresh
                #
                # 原版默认truth_thresh=1，
                # IoU理论上不会 > 1。
                # 这里仍保留完整逻辑。
                # ----------------------------------------
                truth_pos = (best_iou > self.truth_thresh)
                if not truth_pos.any():
                    continue
                pos_idx = truth_pos.nonzero(as_tuple=False)
                for idx in pos_idx:
                    a = int(idx[0])
                    gj = int(idx[1])
                    gi = int(idx[2])

                    t_idx = int(best_gt_idx[a, gj, gi])
                    truth = gt_b[t_idx]
                    cls = int(truth[1])

                    cx = truth[2]
                    cy = truth[3]
                    gw = truth[4]
                    gh = truth[5]

                    gx = cx * W
                    gy = cy * H
                    tx = gx - gi
                    ty = gy - gj

                    global_anchor = (self.masks[level][a])
                    anchor = self.anchors[global_anchor].to(device=raw.device, dtype=raw.dtype)
                    net_w = W * self.strides[level]
                    net_h = H * self.strides[level]

                    tw = torch.log(gw * net_w / anchor[0])
                    th = torch.log(gh * net_h / anchor[1])
                    level_data[level]["positive"][b, a, gj, gi] = True
                    level_data[level]["box_target"][b, a, gj, gi] = torch.stack((tx, ty, tw, th))
                    level_data[level]["box_scale"][b, a, gj, gi] = (2.0 - gw * gh)
                    level_data[level]["cls_target"][b, a, gj, gi, cls] = 1.0

    # ============================================================
    # original best-anchor assignment
    # ============================================================

    def build_positive_from_gt(self, entry, targets, level_data):
        """
        原版Darknet：

        每个GT与全部9个anchor做WH-IoU
                ↓
        best_n = argmax
                ↓
        找该anchor属于哪个YOLO level
                ↓
        唯一positive
        """

        if targets.shape[0] == 0:
            return

        # 网络输入尺寸
        _, _, H5, W5, _ = entry[0].shape
        net_w = (W5 * self.strides[0])
        net_h = (H5 * self.strides[0])
        image_scale = torch.tensor([net_w, net_h], device=entry[0].device, dtype=entry[0].dtype)
        anchors_norm = (self.anchors.to(device=entry[0].device, dtype=entry[0].dtype) / image_scale)

        for truth in targets:
            b = int(truth[0])
            cls = int(truth[1])
            cx = truth[2]
            cy = truth[3]
            gw = truth[4]
            gh = truth[5]
            gt_wh = truth[4:6].reshape(1, 2)

            # ----------------------------------------
            # 与全部9个anchor算WH-IoU
            # ----------------------------------------
            ious = self.wh_iou(gt_wh, anchors_norm)[0]

            # torch.argmax并列时返回第一个，
            # 和Darknet的 ">" 更新逻辑一致
            best_anchor = int(torch.argmax(ious))
            level, local_anchor = (self.anchor_to_level(best_anchor))
            raw = entry[level]
            _, _, H, W, _ = raw.shape
            gx = cx * W
            gy = cy * H
            gi = int(gx)
            gj = int(gy)

            # 正常归一化label应满足0 <= cx,cy < 1
            if not (0 <= gi < W and 0 <= gj < H):
                continue
            tx = gx - gi
            ty = gy - gj
            anchor = self.anchors[best_anchor].to(device=raw.device, dtype=raw.dtype)
            level_net_w = (W * self.strides[level])
            level_net_h = (H * self.strides[level])
            tw = torch.log(gw * level_net_w / anchor[0])
            th = torch.log(gh * level_net_h / anchor[1])

            # ----------------------------------------
            # 原版第二轮GT assignment覆盖前面的状态
            # ----------------------------------------

            level_data[level]["positive"][b, local_anchor, gj, gi] = True
            level_data[level]["box_target"][b, local_anchor, gj, gi] = torch.stack((tx, ty, tw, th))
            level_data[level]["box_scale"][b, local_anchor, gj, gi] = (2.0 - gw * gh)

            # 防止覆盖前先清零
            level_data[level]["cls_target"][b, local_anchor, gj, gi].zero_()
            level_data[level]["cls_target"][b, local_anchor, gj, gi, cls] = 1.0

    # ============================================================
    # use user's all_data2
    # ============================================================

    def build_positive_from_encoded(self, entry, positive_targets, level_data, ):
        """
        使用你现在的all_data2：

        [
            batch,
            cls,
            anchor_idx,
            grid_x,
            grid_y,
            tx,
            ty,
            tw,
            th
        ]
        """

        if positive_targets.shape[0] == 0:
            return

        for t in positive_targets:
            b = int(t[0])
            cls = int(t[1])
            anchor_idx = int(t[2])
            gi = int(t[3])
            gj = int(t[4])
            tx = t[5]
            ty = t[6]
            tw = t[7]
            th = t[8]

            level, local_anchor = (self.anchor_to_level(anchor_idx))
            raw = entry[level]
            _, _, H, W, _ = raw.shape

            if not (0 <= gi < W and 0 <= gj < H):
                continue

            level_data[level]["positive"][b, local_anchor, gj, gi] = True
            level_data[level]["box_target"][b, local_anchor, gj, gi] = torch.stack((tx, ty, tw, th))

            # ----------------------------------------
            # 从 tw/th + anchor反推出GT归一化WH，
            # 用于原版的：
            #
            #   scale = 2 - truth.w * truth.h
            # ----------------------------------------

            anchor = self.anchors[anchor_idx].to(device=raw.device, dtype=raw.dtype)
            net_w = (W * self.strides[level])
            net_h = (H * self.strides[level])
            gw = (torch.exp(tw) * anchor[0] / net_w)
            gh = (torch.exp(th) * anchor[1] / net_h)
            level_data[level]["box_scale"][b, local_anchor, gj, gi] = (2.0 - gw * gh)
            level_data[level]["cls_target"][b, local_anchor, gj, gi].zero_()
            level_data[level]["cls_target"][b, local_anchor, gj, gi, cls] = 1.0

    # ============================================================
    # forward loss
    # ============================================================

    def forward(self, entry, targets, positive_targets=None):
        """
        entry:
            RAW f5,f4,f3

        targets:
            原始GT。

        positive_targets:
            可选，你现在生成的all_data2。
        """

        assert len(entry) == 3

        for x in entry:
            assert x.ndim == 5
            assert x.shape[1] == 3
            assert x.shape[-1] == self.no

        device = entry[0].device
        dtype = entry[0].dtype

        # --------------------------------------------------------
        # 1. 原始GT
        # --------------------------------------------------------

        targets = self.prepare_targets(targets, device, dtype)

        # --------------------------------------------------------
        # 2. 创建dense mask/target
        # --------------------------------------------------------

        level_data = self.make_level_targets(entry)

        # --------------------------------------------------------
        # 3. 第一轮：
        #
        # 对所有prediction与所有GT算IoU
        #
        # negative / ignore / truth_thresh
        # --------------------------------------------------------

        self.build_ignore_and_truth_thresh(entry, targets, level_data)

        # --------------------------------------------------------
        # 4. 第二轮：
        #
        # 原版best-anchor assignment
        #
        # 如果给了你自己的all_data2，
        # 就直接使用all_data2。
        # --------------------------------------------------------

        if positive_targets is None:
            self.build_positive_from_gt(entry, targets, level_data)

        else:
            positive_targets = (self.prepare_positive_targets(positive_targets, device, dtype))
            self.build_positive_from_encoded(entry, positive_targets, level_data)

        # --------------------------------------------------------
        # 5. loss
        # --------------------------------------------------------

        loss_box = entry[0].new_tensor(0.0)
        loss_obj = entry[0].new_tensor(0.0)
        loss_cls = entry[0].new_tensor(0.0)

        num_positive = 0
        num_negative = 0
        num_ignore = 0

        for level, raw in enumerate(entry):
            data = level_data[level]
            pos = data["positive"]
            ignore = data["ignore"]

            # positive优先级高于ignore
            neg = (~pos & ~ignore)
            num_positive += int(pos.sum())
            num_negative += int(neg.sum())
            num_ignore += int((ignore & ~pos).sum())

            # ====================================================
            # bbox
            # ====================================================

            if pos.any():
                raw_box = raw[..., 0:4][pos]
                target_box = data["box_target"][pos]
                scale = data["box_scale"][pos]

                # -----------------------------------------------
                # xy
                #
                # Darknet:
                #
                # delta_x =
                #   scale *
                #   (target_x - sigmoid(raw_x))
                #
                # BCEWithLogits gradient:
                #
                #   sigmoid(raw_x) - target_x
                #
                # 因此梯度等价。
                # -----------------------------------------------

                loss_xy = (
                    F.binary_cross_entropy_with_logits(raw_box[:, 0:2], target_box[:, 0:2], reduction="none").sum(
                        dim=1))

                # -----------------------------------------------
                # wh
                #
                # Darknet:
                #
                # delta_w =
                #   scale *
                #   (target_w - raw_w)
                #
                # 0.5*MSE的梯度正好：
                #
                #   raw_w - target_w
                # -----------------------------------------------

                loss_wh = (0.5 * (raw_box[:, 2:4] - target_box[:, 2:4]).pow(2).sum(im=1))
                loss_box = (loss_box + (scale * (loss_xy + loss_wh)).sum())

                # =================================================
                # class
                # =================================================

                raw_cls = raw[..., 5:][pos]
                target_cls = data["cls_target"][pos]
                loss_cls = (loss_cls + F.binary_cross_entropy_with_logits(raw_cls, target_cls, reduction="sum"))

            # ====================================================
            # objectness positive
            # ====================================================

            raw_obj = raw[..., 4]

            if pos.any():

                loss_obj = (
                    loss_obj +
                    F.binary_cross_entropy_with_logits(raw_obj[pos], torch.ones_like(raw_obj[pos]), reduction="sum")
                )

            # ====================================================
            # objectness negative
            # ====================================================

            if neg.any():

                loss_obj = (
                    loss_obj +
                    F.binary_cross_entropy_with_logits(raw_obj[neg], torch.zeros_like(raw_obj[neg]), reduction="sum")
                )

        total_loss = (loss_box + loss_obj + loss_cls)

        stats = {
            "loss": total_loss.detach(),
            "box": loss_box.detach(),
            "obj": loss_obj.detach(),
            "cls": loss_cls.detach(),

            "positive": num_positive,
            "negative": num_negative,
            "ignore": num_ignore,
        }

        return total_loss, stats