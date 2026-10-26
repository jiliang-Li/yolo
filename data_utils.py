import cv2
import json
import torch
import random
import numpy as np
from yolov3 import Yolov3Head
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
np.set_printoptions(suppress=True, precision=4)
torch.set_printoptions(sci_mode=False, precision=4, edgeitems=20, linewidth=180)


def set_seed(seed: int = 42):
    """
    设置全局随机种子，确保实验可复现
    """
    # 1. 设置 Python 内置 random 模块的种子
    random.seed(seed)
    # 2. 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 3. 设置 PyTorch CPU 的随机种子
    torch.manual_seed(seed)

    # 4. 设置 PyTorch GPU (CUDA) 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子（多卡训练必备）

        # 5. 设置 cuDNN 后端行为
        # deterministic=True: 强制 cuDNN 使用确定性算法，保证相同输入产生相同输出
        torch.backends.cudnn.deterministic = True
        # benchmark=False: 禁用 cuDNN 的自动基准测试（benchmark），因为寻找最快算法的过程是非确定性的
        torch.backends.cudnn.benchmark = False
set_seed(1)


def activate_one(x):
    """
    对应原始Darknet forward_yolo_layer中的激活。
    输入:
        [B, 3, H, W, nc+5]
    输出仍然:
        [B, 3, H, W, nc+5]
    tx, ty:
        sigmoid
    tw, th:
        保持原始值
    obj:
        sigmoid
    cls:
        sigmoid
    """
    y = x.clone()
    # tx, ty
    y[..., 0:2] = torch.sigmoid(x[..., 0:2])
    # tw, th保持raw
    # y[..., 2:4] = x[..., 2:4]
    # objectness + classes
    y[..., 4:] = torch.sigmoid(x[..., 4:])
    return y


def box_iou(b1, b2):
    """
    计算两个边界框的 IoU (Intersection over Union)

    参数:
        b1: Tensor, 形状为 (N, 4) 或 (4,)，格式为 (x1, y1, x2, y2)
        b2: Tensor, 形状为 (M, 4) 或 (4,)，格式为 (x1, y1, x2, y2)

    返回:
        iou: Tensor, 形状为 (N, M)，表示 b1 中每个框与 b2 中每个框的 IoU
    """
    # 确保输入是 Tensor
    if not isinstance(b1, torch.Tensor):
        b1 = torch.tensor(b1, dtype=torch.float32)
    if not isinstance(b2, torch.Tensor):
        b2 = torch.tensor(b2, dtype=torch.float32)

    # 增加维度以支持广播机制，计算所有对的交集
    # b1: (N, 4) -> (N, 1, 4)
    # b2: (M, 4) -> (1, M, 4)
    if b1.dim() == 1:
        b1 = b1.unsqueeze(0)
    if b2.dim() == 1:
        b2 = b2.unsqueeze(0)

    # 计算交集区域的左上角和右下角坐标
    # 左上角取最大值，右下角取最小值
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])  # Left-Top (N, M, 2)
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])  # Right-Bottom (N, M, 2)

    # 计算交集的宽和高，如果无交集则设为 0
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)

    # 计算交集面积
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # 计算两个框各自的面积
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])  # (N,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])  # (M,)

    # 计算并集面积
    union = area1[:, None] + area2[None, :] - inter

    # 计算 IoU，添加 epsilon 防止除以零
    iou = inter / (union + 1e-6)

    return iou.squeeze()


def wh_iou(wh1, wh2):
    """
    wh1: [N, 2]
    wh2: [M, 2]
    """

    wh1 = wh1[:, None, :]   # [N,1,2]
    wh2 = wh2[None, :, :]   # [1,M,2]

    inter = torch.min(wh1, wh2).prod(2)

    area1 = wh1.prod(2)
    area2 = wh2.prod(2)

    return inter / (area1 + area2 - inter)


anchors = torch.tensor([
    [10, 13],      # 0
    [16, 30],      # 1
    [33, 23],      # 2

    [30, 61],      # 3
    [62, 45],      # 4
    [59, 119],     # 5

    [116, 90],     # 6
    [156, 198],    # 7
    [373, 326],    # 8
])
scale = [32, 16, 8]
grid = [13, 26, 52]

nc = 2
classes = ["person", "horse"]
build_map = [torch.zeros((3, 13, 13, 7)), torch.zeros((3, 26, 26, 7)), torch.zeros((3, 52, 52, 7))]
all_obj_all_data = []
p5, p4, p3 = torch.randn((1, 1024, 13, 13)), torch.randn((1, 512, 26, 26)), torch.randn((1, 256, 52, 52))
head = Yolov3Head(n_classes=2)
f5, f4, f3 = head((p5, p4, p3))
f5, f4, f3 = activate_one(f5), activate_one(f4), activate_one(f3)
print(f"f5={f5.shape}, f4={f4.shape}, f3={f3.shape}")

im_sz = 416

img = cv2.imread("./dataset/images/train/000009.jpg")
h, w, _ = img.shape

with open("./dataset/labels/000009.json", "r") as f:
    data = json.load(f)["objects"]
    for obj in data:
        obj[0] = classes.index(obj[0])
data = np.array(data, dtype=np.float32)     # cls, x1, y1, x2, y2
data2 = data.copy()
data2[:, [3, 4]] -= data2[:, [1, 2]]
data2[:, [1, 2]] += (data2[:, [3, 4]] / 2)  # cls, cx, cy, w, h
data2[:, [1, 3]] /= w
data2[:, [2, 4]] /= h


def test2():
    global build_map

    for obj in data2:
        obj_all_data = []
        cls, norm_cx, norm_cy, norm_w, norm_h = obj     # 现在是归一化后的位置信息
        cls_one_hot = F.one_hot(torch.tensor(int(cls)), num_classes=nc).float()
        ocx, ocy, ow, oh = obj[1:] * im_sz      # norm_* * im_sz
        for anchor_i in range(9):
            scale_i = 2 - anchor_i // 3     # 9个anc分3个一组得到012，再与2取反得到210
            anchor = anchors[anchor_i]
            norm_anchor = anchor / im_sz

            # 计算gt中心点位置
            gx = norm_cx * grid[scale_i]
            gy = norm_cy * grid[scale_i]
            index_x = int(gx)
            index_y = int(gy)

            # 计算gt在这个点位上的偏移
            offset_x = gx - index_x     # 偏移位置在这个点的感受视野上的归一化值
            offset_y = gy - index_y
            offset_ow = torch.log(ow / anchor[0]).item()   # 必须取log,加log主要可以使gt和anchor的w\h在某一区间内对称起来
            offset_oh = torch.log(oh / anchor[1]).item()

            # 计算gt和anchor都至于中心点[0, 0]时的iou
            iou = wh_iou(torch.from_numpy(np.stack([norm_w, norm_h]))[None], norm_anchor[None]).item()
            # print(iou)

            obj_all_data.append(
                torch.cat([
                    torch.tensor([iou, anchor_i, index_y, index_x, offset_x, offset_y, offset_ow, offset_oh, 0]),
                    cls_one_hot
                ])
            )
        obj_all_data = torch.stack(obj_all_data, dim=0)
        best_iou = obj_all_data[:, 0].max()
        # obj_all_data[:, 8] = obj_all_data[:, 0] == best_iou   # 并列最大时设置多个
        for obj_data in obj_all_data:   # 只设置一个最大值
            if obj_data[0] == best_iou:
                obj_data[8] = 1
                all_obj_all_data.append(obj_data)
                break
        # print(obj_all_data)
        for obj_data in obj_all_data:
            # iou, anc_i, y_i, x_i, x, y, w, h, conf, cls0, cls1
            scale_i = 2 - int(obj_data[1] / 3)
            build_map[scale_i][int(obj_data[1] % 3), int(obj_data[2]), int(obj_data[3])] = obj_data[4:]
        # break


# test2()
# all_obj_all_data = torch.stack(all_obj_all_data, dim=0)
# print(all_obj_all_data)

def test3():
    global build_map, all_obj_all_data
    all_obj_norm = torch.from_numpy(data2)
    all_anc_norm = anchors / im_sz
    all_iou = wh_iou(all_obj_norm[:, [3, 4]], all_anc_norm)
    col_idx = torch.argmax(all_iou, dim=1)      # anchor idx
    row_idx = torch.arange(all_iou.size(0))     # object idx
    best_iou = all_iou[row_idx, col_idx]

    # cls, norm[cx, cy, w, h], best_iou, anc_idx
    all_data = torch.cat((all_obj_norm[row_idx], best_iou.reshape(-1, 1), col_idx.reshape(-1, 1)), dim=1)
    all_data2 = torch.zeros(all_data.size())
    print(all_data)

    scale_idx = 2 - col_idx // 3
    grids = torch.tensor(grid)[scale_idx].reshape(-1, 1)
    all_data2[:, [0, 1]] = all_data[:, [1, 2]] * grids
    all_data2[:, [2, 3]] = all_data2[:, [0, 1]] - all_data2[:, [0, 1]].int()    # offset_x, offset_y
    all_data2[:, [0, 1]] = all_data2[:, [0, 1]].int().float()                   # col_idx, row_idx
    all_data2[:, [4, 5]] = torch.log(all_data[:, [3, 4]] / all_anc_norm[col_idx])   # log(ow/anc_w)
    all_data2[:, 6] = 1

    # cls, anchor_idx, grid_x, grid_y, offset_x, offset_y, offset_w, offset_h
    all_data2 = torch.cat((all_data[:, 0:1], all_data[:, -1:], all_data2[:, :-1]), dim=1)
    print(all_data2)


# test3()


def test4():
    imgs = [
        "./dataset/images/train/000009.jpg",
        ".\\dataset\\images\\train\\000017.jpg"
    ]
    labels = ".\\dataset\\labels\\000009.json", ".\\dataset\\labels\\000017.json"

    for i in range(2):
        img = cv2.imread(imgs[i])
        h, w, c = img.shape
        with open(labels[i], "r") as f:
            data = json.load(f)["objects"]
            for obj in data:
                obj[0] = classes.index(obj[0])
        data = np.array(data, dtype=np.float32) # cls xyxy
        data[:, [3, 4]] -= data[:, [1, 2]]      # cls xywh
        data[:, [1, 2]] += data[:, [3, 4]] / 2.0       # cls cx cy w h
        data[:, [1, 3]] /= w
        data[:, [2, 4]] /= h
        data = data.tolist()
        data = ["%d %.6f %.6f %.6f %.6f" % tuple(obj) for obj in data]
        print(data)
        with open(imgs[i].replace('images', 'labels').replace('.jpg', '.txt'), 'w') as fwrite:
            fwrite.write('\n'.join(data))
test4()