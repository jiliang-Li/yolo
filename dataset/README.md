# YOLOv3 数据加载

`dataset.py` 提供 PyTorch `class Dataset`，只加载和预处理，不做随机裁剪、翻转、颜色扰动或多尺度增强。

```text
data/
  images/train/001.jpg
  labels/train/001.txt
  train.txt
```

标签文本每行：`class_id x_center y_center width height`。类别从 0 开始，坐标相对**原图**宽高归一化。无目标图片使用空标签文件；缺失标签文件会报错。

```python
from torch.utils.data import DataLoader
from dataset import Dataset

dataset = Dataset("data/images/train", img_size=416, num_classes=80)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
images, targets = next(iter(loader))
# images:  [B, 3, 416, 416], float32 RGB, [0, 1]
# targets: [B, 90, 5], float32，每行为 [x, y, w, h, class_id]
valid = (targets[..., 2:4] > 0).all(dim=-1)
```

也支持 `Dataset("data/train.txt")`，每行一个图片路径。相对路径以列表文件所在目录为基准，例如上面的 `train.txt` 可写 `images/train/001.jpg`。若已有列表按项目根目录写路径，请通过 `root="项目根目录"` 指定基准。

图片路径中的 `images`、`JPEGImages` 或 `raw` 目录映射为 `labels`，保留子目录和文件名，扩展名改成 `.txt`；没有这些目录名时读取图片旁的同名 `.txt`。

预处理采用 Darknet 的确定性 `letterbox_image`：等比例双线性缩放（端点对齐）、缩放尺寸向下取整、居中填充 **0.5**。标签根据实际缩放尺寸和左上填充量同步转换，坐标仍相对最终输入归一化；框边界裁剪到输入范围，宽或高小于 0.001 的框被过滤。极端长宽比导致短边不足一个像素时，将短边保留为 1 像素。

注意原版训练加载器本身包含随机变换；本实现按无增强要求采用原版推理的居中 letterbox。标签文件是 `class x y w h`，返回值则采用 Darknet 内部 truth 的 `x y w h class` 顺序，默认 90 行、零填充。超过 `max_boxes` 时发出警告并按文件顺序截取，再过滤小框。没有预先分配三个尺度的网格或 anchor，这部分属于 YOLO loss 的职责。需要原版扁平 truth 时，可使用 `targets.flatten(1)`，得到 `[B, 450]`。

参考原版实现：[图像预处理](https://github.com/pjreddie/darknet/blob/master/src/image.c)、[标签加载](https://github.com/pjreddie/darknet/blob/master/src/data.c)、[YOLO truth 结构](https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c)。

安装依赖：`python -m pip install -r requirements.txt`。运行验证：`python -m unittest discover -s tests -v`。

> **每张图片的标签默认是 `[90, 5]`**，每行表示一个目标框：`[x, y, w, h, class_id]`。
> 
> 例如一张图片有 3 个目标：
> - 前 3 行存放真实标签。
> - 后 87 行全部填 `0`。
> - 超过 90 个目标时，当前实现会发出警告，只保留前 90 行，再过滤过小的框。
>
> 这样一个 batch 就能直接堆叠成 `[batch_size, 90, 5]`。90 是容量上限，不代表每张图片实际有 90 个目标，可以通过 `max_boxes` 修改。
> 
> 判断有效标签应看宽高是否大于 0，不能用 `class_id != 0`，因为类别 0 也是合法类别。