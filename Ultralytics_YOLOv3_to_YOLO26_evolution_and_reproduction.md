# Ultralytics YOLO：从 YOLOv3 到 YOLO26 的发展历程、创新点与逐组件复现路线

> 目标：不是只会调用 `ultralytics.YOLO()`，而是沿着 YOLO 的演进路线，从一个可工作的 YOLOv3 基线开始，逐版本、逐组件地替换和复现关键设计，理解每次改动解决了什么问题、如何实现、收益来自哪里、又付出了什么代价。

---

## 0. 范围说明

本文只沿着 **Ultralytics 当前官方架构文档明确梳理的主线**展开：

```text
YOLOv3 → YOLOv5 → YOLOv8 → YOLO11 → YOLO26
```

需要注意：

- **YOLOv3 原始模型不是 Ultralytics 提出的**，原作者为 Joseph Redmon 和 Ali Farhadi；Ultralytics 后续维护了 YOLOv3 的 PyTorch 实现，并在当前文档中把 YOLOv3 作为其架构演进起点。
- YOLOv4、YOLOv6、YOLOv7、YOLOv9、YOLOv10、YOLOv12 等虽然同属 YOLO 家族，且部分模型在 Ultralytics 文档中有介绍/支持，但不是本文要追踪的 Ultralytics 主线迭代。
- 对 YOLOv3 / YOLOv5，本文主要讨论它们**历史上的原始 anchor-based 设计**；当前 `ultralytics` 包中的 `YOLOv3u/YOLOv5u` 已换成类似 YOLOv8 的 anchor-free head，不能把两者混为一谈。

---

# 1. 一张表看完整演进

| 版本 | 核心 Backbone 模块 | Neck | Detect Head | Anchor | Objectness | Box 表达/回归 | Label Assignment | 主要 Loss | NMS | 最重要的结构变化 |
|---|---|---|---|---|---|---|---|---|---|---|
| **YOLOv3** | Darknet-53 / Bottleneck / Residual | FPN，仅 Top-down | Coupled Head | 是 | 是 | Anchor offset | Anchor matching | Ultralytics 实现：CIoU + BCE Obj + BCE Cls | 需要 | Darknet-53、Residual、FPN、P3/P4/P5 三尺度检测 |
| **YOLOv5** | CSPDarknet / C3 | FPN + PAN + SPPF | YOLOv3-style Coupled Head | 是 | 是 | 改进 Anchor offset | Anchor ratio + 邻格扩展 | CIoU + BCE Obj + BCE Cls | 需要 | CSP/C3、PAN、SPPF、改进 bbox 参数化、成熟训练工程 |
| **YOLOv8** | C2f | FPN + PAN + SPPF | Anchor-free Decoupled Head | 否 | 否 | LTRB distance distribution | TaskAlignedAssigner | IoU + DFL + BCE Cls | 需要 | Anchor-free、Objectness-free、Decoupled Head、DFL、TAL |
| **YOLO11** | C3k2 | FPN + PAN + SPPF + C2PSA | Anchor-free Decoupled Head | 否 | 否 | LTRB + DFL | TaskAlignedAssigner | IoU + DFL + BCE Cls | 需要 | C3k2、C2PSA、轻量分类分支 |
| **YOLO26** | C3k2 | FPN + PAN，SPPF residual，C2PSA | Dual Head：O2M + O2O | 否 | 否 | 直接 distance regression，DFL-free | STAL / O2M + O2O | IoU + Cls + normalized distance L1 | O2M 需要；O2O 可 NMS-free | End-to-End、NMS-free、DFL-free、Dual Head、STAL、Progressive Loss、MuSGD |

---

# 2. 整条发展路线到底在解决什么

如果只记模块名，很容易把 YOLO 学成“背版本”。更值得关注的是它一直在解决以下几个问题。

## 2.1 Backbone：怎样提取更强特征，同时减少重复计算

```text
YOLOv3        YOLOv5        YOLOv8        YOLO11/26
Bottleneck -> C3/CSP     -> C2f        -> C3k2
Residual      Partial       更多中间      更灵活的 CSP
堆叠          gradient       特征复用      内部单元
```

总体趋势：

```text
更深
  ↓
避免重复计算
  ↓
改善梯度流
  ↓
提高特征复用
  ↓
在精度、参数量、FLOPs之间重新平衡
```

---

## 2.2 Neck：怎样让高层语义和低层定位信息充分融合

YOLOv3：

```text
P5
 ↓  top-down
P4
 ↓
P3
```

只有 FPN 式 Top-down 路径。

YOLOv5 开始加入 PAN Bottom-up：

```text
       FPN
P5 → P4 → P3
          ↓
       PAN
P3 → P4 → P5
```

此后 YOLOv8 / YOLO11 / YOLO26 基本延续这类双向多尺度融合框架，重点转向融合位置使用什么 block。

---

## 2.3 Head：从“Anchor + Objectness”走向 Anchor-free

经典 YOLOv3/v5：

```text
grid cell
  ├─ anchor 1 → [x,y,w,h,obj,classes]
  ├─ anchor 2 → [x,y,w,h,obj,classes]
  └─ anchor 3 → [x,y,w,h,obj,classes]
```

YOLOv8/11：

```text
anchor point
   ├─────────────┐
   ▼             ▼
box branch    cls branch
LTRB distance   classes
```

发生了三件本质变化：

1. 去掉预定义 anchor shape；
2. 去掉独立 objectness；
3. bbox 与 classification 使用独立分支。

---

## 2.4 Label Assignment：从“哪个 Anchor 像 GT”到“哪个预测最适合 GT”

```text
YOLOv3
Anchor matching
      ↓
YOLOv5
Anchor ratio + neighbor-grid expansion
      ↓
YOLOv8 / YOLO11
TaskAlignedAssigner
classification × localization quality
      ↓
YOLO26
O2M + O2O + Small-Target-Aware Label Assignment
```

这是整个 YOLO 演化中最值得深入复现的一条线。

早期关注的是：

> GT 的宽高和哪个 anchor 最接近？

现代检测器更关注：

> 当前网络的哪些候选点，在“类别判断”和“定位质量”上最适合负责这个 GT？

到了端到端检测，还要解决：

> 如果一个 GT 最终只能对应一个预测，怎样避免正样本太少以及小目标分不到合适正样本？

---

## 2.5 Box Regression：从参数化 Anchor 到 Distribution，再回到直接距离回归

### YOLOv3 原始思想

```text
anchor
  +
predicted offset
  ↓
bbox
```

### YOLOv8 / YOLO11

不再预测 anchor offset，而是基于 anchor point 预测：

```text
left
 top
right
bottom
```

每条边又不是一个单值，而是一个离散分布：

```text
0, 1, 2, ..., 15
```

即 `reg_max=16`，通过 DFL 训练。

### YOLO26

又改为：

```text
reg_max = 1
```

直接回归距离，DFL 模块退化为 Identity。

这不是“回到 YOLOv5”，因为其整体仍然是 Anchor-free + distance regression，只是删除了分布式 bbox 表示。

---

## 2.6 后处理：从 Dense Prediction + NMS 到 One-to-One End-to-End

YOLOv3 → YOLO11：

```text
一个 GT
 ↓
训练出多个高分候选框
 ↓
NMS
 ↓
保留最终框
```

YOLO26 增加 One-to-One Head：

```text
一个 GT
 ↓
训练目标就是让一个预测负责
 ↓
直接输出
 ↓
无需 NMS
```

因此 YOLO26 同时维护两套逻辑：

```text
One-to-Many Head
训练监督充足 / 默认推理使用 NMS

One-to-One Head
面向端到端推理 / 可 nms=False
```

---

# 3. YOLOv3：建立基线

## 3.1 需要掌握的组件

- Conv + BN + activation
- Residual Bottleneck
- Darknet-53
- P3 / P4 / P5 feature maps
- FPN Top-down
- Upsample + Concat
- Anchor
- Grid
- Coupled Detect Head
- bbox decode
- Objectness
- Multi-label BCE classification
- IoU / CIoU
- NMS
- Anchor-based target assignment

## 3.2 YOLOv3 最值得复现的核心

### A. Darknet-53 + Residual

理解：

```text
x ──────────────┐
                + → output
x → Conv → Conv ┘
```

复现时验证：

- 输入输出 tensor shape；
- shortcut 条件；
- stride=2 下采样的位置；
- P3/P4/P5 的 stride 是否分别为 8/16/32。

### B. FPN

需要亲自实现：

```text
P5 → Conv → Upsample → Concat(P4)
                         ↓
                       feature
                         ↓
             Upsample → Concat(P3)
```

重点理解：

- 为什么高层 feature 语义强；
- 为什么浅层 feature 空间信息丰富；
- 为什么 concat 后还需要卷积融合。

### C. Anchor-based Detect Head

需要从 tensor 维度开始推：

若：

```text
na = 3
nc = 80
```

则每层输出通道：

```text
3 × (5 + 80) = 255
```

拆分为：

```text
[x, y, w, h, obj, cls_1 ... cls_80]
```

### D. bbox decode

至少自己实现并验证：

- grid 构造；
- anchor scale；
- xy decode；
- wh decode；
- grid 坐标 → input image 坐标。

### E. Target Assignment

这是第一阶段最重要的训练逻辑。

建议自己打印：

```text
GT id
→ detection level
→ anchor id
→ grid_x/grid_y
→ positive target
```

不要只看最终 loss。

### F. Loss

对于 Ultralytics YOLOv3 legacy 实现，可拆成：

```text
L = λbox * Lbox
  + λobj * Lobj
  + λcls * Lcls
```

其中：

- `Lbox`：CIoU；
- `Lobj`：BCE；
- `Lcls`：BCE；
- box/cls 只对 positive 计算；
- objectness 对所有候选位置计算。

## 3.3 YOLOv3 阶段验收标准

你应该可以不看源码解释清楚：

- [ ] 一张 640×640 图片为什么会生成多个尺度的预测；
- [ ] 每个 grid 为什么需要多个 anchor；
- [ ] 一个 GT 怎样找到负责它的预测位置；
- [ ] 为什么负样本通常只算 objectness loss；
- [ ] bbox 的 raw output 怎样还原成实际坐标；
- [ ] 为什么需要 NMS；
- [ ] P3 为什么更适合小目标，P5 为什么更适合大目标。

---

# 4. YOLOv3 → YOLOv5：先不要换检测范式，只优化经典 YOLO

YOLOv5 非常适合做第一轮“单变量结构升级”。

## 4.1 建议的复现顺序

```text
YOLOv3 baseline
    ↓
① Bottleneck stack → CSP/C3
    ↓
② FPN → FPN + PAN
    ↓
③ SPP → SPPF / 加入 SPPF
    ↓
④ bbox decode 改进
    ↓
⑤ neighbor-grid target assignment
    ↓
⑥ AutoAnchor / augmentation / EMA / AMP 等训练工程
    ↓
YOLOv5-like
```

不要一次全换，否则无法判断收益来源。

---

## 4.2 C3 / CSP

### 要回答的问题

Darknet-53 直接堆 Bottleneck 有什么重复计算？

CSP 思路：

```text
input
 ├──────────── bypass ──────────────┐
 │                                   │
 └→ conv → Bottleneck × N ──────────┤
                                     ↓
                                   concat
                                     ↓
                                    conv
```

复现实验：

```text
Experiment V5-01
YOLOv3 backbone
vs
replace Bottleneck stack with C3
```

记录：

- Params；
- GFLOPs；
- peak VRAM；
- epoch time；
- mAP；
- 收敛曲线。

---

## 4.3 PAN

YOLOv3：

```text
P5 → P4 → P3
```

加入：

```text
P3 → P4 → P5
```

实验：

```text
FPN only
vs
FPN + PAN
```

重点观察：

- small AP；
- medium AP；
- large AP；
- 中高层 feature 是否能重新获得定位信息。

---

## 4.4 SPPF

SPPF 使用连续 5×5 MaxPool：

```text
x
├──────────────┐
├→ pool5 ──────┤
├→ pool5→pool5 ┤
└→ pool5→pool5→pool5
               ↓
             concat
               ↓
              conv
```

其有效感受野对应经典 SPP 的 5/9/13 kernel，但计算更高效。

需要验证：

- 输出 shape 不变；
- receptive field 增大；
- 与并行 SPP 的运行时间比较。

---

## 4.5 bbox 参数化改进

Ultralytics YOLOv5 典型方式：

```text
xy = sigmoid(raw_xy) * 2 - 0.5
wh = (sigmoid(raw_wh) * 2)^2 * anchor
```

相比早期严格限制在 `(0,1)` 的 center offset，xy 范围扩展到：

```text
(-0.5, 1.5)
```

重点观察 grid boundary 附近目标。

---

# 5. YOLOv5 → YOLOv8：第一次大的检测范式变化

这一阶段不应该只做“换 backbone”。建议拆成多个独立实验。

## 5.1 推荐顺序

```text
YOLOv5-like
   ↓
① C3 → C2f
   ↓
② Coupled → Decoupled Head
   ↓
③ Anchor-based → Anchor-free
   ↓
④ 删除 Objectness
   ↓
⑤ Anchor matching → TaskAlignedAssigner
   ↓
⑥ scalar box regression → DFL distribution regression
   ↓
YOLOv8-like
```

---

## 5.2 C2f

C3 最终只把两路主要 feature concat。

C2f 更强调中间 Bottleneck 输出复用：

```text
input
 ↓
Conv → split
       ├────────────┐
       └→ B1 → B2 → B3 ...
           │    │    │
           └────┴────┴──→ concat all
                           ↓
                          Conv
```

实验重点：

- gradient flow；
- 中间特征复用；
- 参数量/FLOPs；
- 同等训练设置下的收敛。

---

## 5.3 Decoupled Head

YOLOv5：

```text
shared feature
      ↓
   one branch
      ↓
box + obj + cls
```

YOLOv8：

```text
shared feature
 ├─────────────┐
 ▼             ▼
box branch   cls branch
```

复现时最好检查两个 branch 的梯度：

```python
box_loss.backward(retain_graph=True)
# 观察 box branch / cls branch gradient
```

目的不是实际这么训练，而是理解任务解耦后梯度怎样流动。

---

## 5.4 Anchor-free

删除：

```text
anchor width
anchor height
anchor clustering
AutoAnchor
```

保留 feature-map point：

```text
point (cx, cy)
```

然后预测：

```text
l, t, r, b
```

框恢复：

```text
x1 = cx - l
x2 = cx + r
y1 = cy - t
y2 = cy + b
```

这个实验一定要从几何关系自己写，不建议先复制 Ultralytics 的 decode 函数。

---

## 5.5 删除 Objectness

经典：

```text
score = objectness × class score
```

现代 Ultralytics YOLO：

```text
box branch + class branch
```

不再额外输出 objectness channel。

需要理解：

- label assignment 已经把 classification quality 与 localization quality 更深地绑定；
- classification target/score 不再只是纯类别概率意义；
- 少一个独立预测任务也简化了 head。

---

## 5.6 TaskAlignedAssigner

这是 YOLOv8 最值得重点复现的算法之一。

建议不要先完整照抄，而先实现一个简化版本：

```text
candidate points
       ↓
point inside GT ?
       ↓
classification score
       ×
IoU score
       ↓
alignment metric
       ↓
Top-K
       ↓
positive samples
```

然后再与官方实现逐 tensor 对齐。

重点记录：

```text
每个 GT 正样本数
正样本位于哪些 level
小/中/大目标正样本分布
训练早期 vs 后期 assignment 是否变化
```

---

## 5.7 DFL

YOLOv8 / YOLO11：

```text
reg_max = 16
```

一条边不是输出：

```text
7.4
```

而是输出 16 个 logits：

```text
P(0), P(1), ..., P(15)
```

softmax 后取期望：

```text
d = Σ i * P(i)
```

四条边因此需要：

```text
4 × 16 = 64
```

个 bbox regression channel。

复现步骤：

1. 单独实现 Distribution → scalar；
2. 构造人工 target，例如 `7.4`；
3. 理解它为什么对相邻 bin 同时监督；
4. 再接入 Detect Head；
5. 比较直接 L1/IoU regression 与 DFL 的收敛差异。

---

# 6. YOLOv8 → YOLO11：在现代检测范式下继续提效

YOLO11 没有重新发明 detection paradigm。

基本仍然是：

```text
Anchor-free
Decoupled Head
TaskAlignedAssigner
DFL
IoU loss
NMS
```

重点变成 feature extractor 与 head efficiency。

---

## 6.1 C2f → C3k2

建议直接做模块级对比：

```text
C2f
vs
C3k2
```

统一：

- input channels；
- output channels；
- repeats；
- feature-map size。

记录：

```text
Params
MACs/FLOPs
latency
activation memory
output statistics
```

---

## 6.2 C2PSA

YOLO11 在 SPPF 后加入 C2PSA。

可概括为：

```text
feature
 ↓
CSP split
 ├──────────── bypass
 └→ PSABlock × N
       ├→ multi-head attention
       └→ FFN
 ↓
concat
 ↓
fusion conv
```

建议复现重点：

- attention 输入 shape；
- Q/K/V shape；
- spatial token 数；
- residual；
- FFN；
- attention 带来的显存和 latency 成本。

---

## 6.3 Lightweight Classification Branch

YOLO11/YOLO26 的 classification branch 使用更轻量的 depthwise-separable block。

实验：

```text
普通 3×3 Conv classification branch
vs
DWConv + 1×1 Conv
```

比较：

- 参数量；
- FLOPs；
- GPU latency；
- CPU latency；
- cls AP 是否受影响。

---

# 7. YOLO11 → YOLO26：端到端检测

这是第二次大的范式变化。

## 7.1 推荐复现顺序

```text
YOLO11-like
   ↓
① SPPF residual shortcut
   ↓
② reg_max 16 → 1，移除 DFL
   ↓
③ 直接 distance regression + normalized L1
   ↓
④ 增加 One-to-One Head
   ↓
⑤ Dual-head training：O2M + O2O
   ↓
⑥ NMS-free inference
   ↓
⑦ STAL
   ↓
⑧ Progressive Loss
   ↓
⑨ MuSGD
   ↓
YOLO26-like
```

---

## 7.2 SPPF Residual

YOLO26 保留 SPPF，但 deepest SPPF 增加 shortcut：

```text
x ─────────────────────┐
                       + → y
x → SPPF transformations
```

这是非常适合做第一个 YOLO26 单变量实验的改动。

---

## 7.3 DFL-free

YOLO11：

```text
reg_max = 16
4 × 16 bbox channels
DFL decode
```

YOLO26：

```text
reg_max = 1
4 bbox channels
DFL = Identity
```

收益：

- head 更轻；
- export graph 更简单；
- CPU/边缘设备友好；
- 不再需要 distribution integral。

代价：

- 删除 DFL 后，需要重新设计 bbox supervision 保证定位质量。

官方 YOLO26 training recipe 中虽然仍保留名为 `dfl` 的 gain，但其作用已经变为 **normalized box-distance L1 loss 的权重**，不再是 Distribution Focal Loss。

---

## 7.4 One-to-Many 与 One-to-One Dual Head

YOLO26 最重要的部分。

```text
                   shared features
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
     One-to-Many Head        One-to-One Head
           │                       │
     多个 positive / GT       单一匹配目标
           │                       │
     supervision richer       end-to-end inference
```

训练：两个 head 都接受监督。

推理：

```text
默认：O2M → NMS
nms=False：O2O → no NMS
```

需要自己验证：

- 同一个 GT 在 O2M 中对应多少 positive；
- 在 O2O 中最终对应多少 positive；
- 两个 branch 是否共享 feature 但拥有独立 Detect parameters；
- NMS-free 结果中 duplicate prediction 是否明显减少。

---

## 7.5 STAL：Small-Target-Aware Label Assignment

目标：改善小目标在稀疏/严格 assignment 中正样本覆盖不足的问题。

建议研究问题：

```text
small GT
 ↓
在 P3 上覆盖多少 candidate points？
 ↓
Top-K 后剩几个？
 ↓
One-to-One 是否有稳定 positive？
```

实验时不要只看 overall mAP，单独记录：

```text
AP_small
small-GT positive count
positive coverage ratio
```

---

## 7.6 Progressive Loss

核心思想可理解为：

> 训练过程不让 O2M 与 O2O 永远保持完全相同的监督权重，而是逐步把优化重点转向最终推理真正使用的端到端分支。

复现时建议先实现简单版本：

```text
L = λ_o2m(epoch) * L_o2m
  + λ_o2o(epoch) * L_o2o
```

然后绘图：

```text
x-axis = epoch
y-axis = λ
```

观察不同 schedule 对：

- 收敛速度；
- O2M AP；
- O2O AP；
- duplicate prediction；
- small AP

的影响。

---

## 7.7 MuSGD

官方 YOLO26 checkpoint 使用 MuSGD：将传统 SGD 与 Muon-style 的矩阵更新结合。

建议把优化器放在最后复现。

原因：

```text
网络结构
label assignment
loss
```

如果这些都没对齐，先换 optimizer 很难判断收益来源。

推荐顺序：

```text
SGD baseline
 ↓
AdamW baseline（可选）
 ↓
Muon-style update demo
 ↓
MuSGD
```

重点比较：

- loss convergence；
- gradient norm；
- weight norm；
- final mAP；
- optimizer state memory；
- training throughput。

---

# 8. 推荐的完整逐版本复现路线

## Phase 0：搭一个完全属于自己的最小检测框架

不要一开始依赖 Ultralytics Trainer。

至少自己拥有：

```text
Dataset
Dataloader
Augmentation
Model
Loss
Target Assignment
Optimizer
Train Loop
EMA（后加）
Validation
mAP
NMS
Checkpoint
```

目录可以设计为：

```text
yolo_rebuild/
├── configs/
├── datasets/
├── models/
│   ├── common.py
│   ├── yolo_v3.py
│   ├── yolo_v5.py
│   ├── yolo_v8.py
│   ├── yolo11.py
│   └── yolo26.py
├── modules/
│   ├── conv.py
│   ├── bottleneck.py
│   ├── c3.py
│   ├── c2f.py
│   ├── c3k2.py
│   ├── sppf.py
│   ├── attention.py
│   └── head.py
├── assigners/
│   ├── anchor_assigner.py
│   ├── task_aligned.py
│   └── stal.py
├── losses/
│   ├── iou.py
│   ├── yolo_v3_loss.py
│   ├── dfl.py
│   └── yolo26_loss.py
├── utils/
│   ├── boxes.py
│   ├── nms.py
│   ├── metrics.py
│   └── visualize.py
├── train.py
└── val.py
```

---

# 9. 推荐的实验编号

建议以后所有代码和实验都按编号保存。

| ID | 改动 | 目的 |
|---|---|---|
| V3-00 | 最小 YOLOv3 baseline | 建立基准 |
| V3-01 | Darknet-53 residual | 验证 backbone |
| V3-02 | FPN P3/P4/P5 | 验证多尺度 |
| V3-03 | Anchor matching | 理解 target |
| V3-04 | CIoU + Obj + Cls | 完整训练 |
| V5-01 | Bottleneck → C3 | CSP 收益 |
| V5-02 | FPN → FPN+PAN | 双向融合 |
| V5-03 | +SPPF | receptive field / efficiency |
| V5-04 | bbox decode 改进 | grid sensitivity |
| V5-05 | neighbor assignment | 增加 positive |
| V8-01 | C3 → C2f | gradient/feature reuse |
| V8-02 | Coupled → Decoupled | task conflict |
| V8-03 | Anchor → Anchor-free | 去 anchor |
| V8-04 | remove objectness | 简化 head |
| V8-05 | TaskAlignedAssigner | dynamic assignment |
| V8-06 | +DFL | distribution regression |
| V11-01 | C2f → C3k2 | backbone efficiency |
| V11-02 | +C2PSA | attention |
| V11-03 | lightweight cls head | head efficiency |
| V26-01 | SPPF shortcut | residual pooling block |
| V26-02 | reg_max 16 → 1 | DFL-free |
| V26-03 | normalized distance L1 | 补偿 direct regression |
| V26-04 | +One-to-One Head | end-to-end 基础 |
| V26-05 | Dual-head training | O2M/O2O 协同 |
| V26-06 | NMS-free inference | 删除后处理 |
| V26-07 | +STAL | small-object assignment |
| V26-08 | Progressive Loss | supervision schedule |
| V26-09 | MuSGD | optimizer 演进 |

---

# 10. 每个实验必须记录什么

不要只记录 mAP。

## 10.1 模型指标

```text
Params
GFLOPs
model size
activation memory
peak VRAM
```

## 10.2 训练指标

```text
box loss
obj loss（若有）
cls loss
DFL / distance loss
learning rate
gradient norm
epoch time
samples/sec
```

## 10.3 检测指标

```text
mAP50
mAP50-95
precision
recall
AP_small
AP_medium
AP_large
```

## 10.4 推理指标

```text
preprocess latency
network latency
NMS latency
end-to-end latency
CPU latency
GPU latency
```

## 10.5 Assignment 指标

强烈建议额外记录：

```text
positive samples / image
positive samples / GT
positive samples / P3
positive samples / P4
positive samples / P5
small-object positive count
medium-object positive count
large-object positive count
```

这是理解检测器非常有效的一组 debug 数据。

---

# 11. 每次改一个组件时的记录模板

后续每一个功能组件都建议按照下面这个模板写笔记。

```markdown
## Component: C2f

### 1. 它替换了什么
C3

### 2. 上一版本存在什么问题
...

### 3. 核心结构
...

### 4. 数学/算法原理
...

### 5. Tensor shape
input : [B, C, H, W]
...
output: [B, C2, H, W]

### 6. 从零实现
...

### 7. 与 Ultralytics 源码逐行对齐
...

### 8. 单元测试
...

### 9. Ablation
baseline:
new:

### 10. 收益
...

### 11. 代价
...

### 12. 我的结论
...
```

---

# 12. 复现时的几个原则

## 原则 1：先正确，再快

第一版优先：

```text
代码简单
shape 清楚
容易 print
容易画图
容易 debug
```

不要一开始追求 fused kernel、TensorRT、CUDA extension。

---

## 原则 2：先写最简算法，再对源码

例如 TaskAlignedAssigner：

错误学习方式：

```text
复制官方 200 行代码
→ 能运行
→ 不知道为什么
```

推荐方式：

```text
先自己写 30 行简化 TAL
→ 人工 tensor 测试
→ 可视化 positive
→ 再读官方实现
→ 对齐 edge cases
```

---

## 原则 3：每次只改变一个主要变量

例如研究 PAN：

```text
Baseline = YOLOv3
Experiment = YOLOv3 + PAN
```

不要同时：

```text
+PAN
+C3
+SPPF
+Mosaic
+新的 optimizer
```

否则结果无法归因。

---

## 原则 4：固定一套长期使用的数据集

建议分两层：

### Debug Dataset

非常小，只用于验证代码：

```text
几十～几百张
训练几分钟能过拟合
```

要求模型能够很快：

```text
train loss → 很低
prediction → 基本正确
```

它用于发现：

- target assignment 错误；
- bbox decode 错误；
- label scale 错误；
- loss 没有梯度；
- NMS 坐标错误。

### Benchmark Dataset

固定一个中等规模数据集，所有版本都在完全相同的 train/val split 上测试。

这样才能比较：

```text
V3-00
V5-01
V5-02
...
V26-09
```

---

## 原则 5：必须可视化 Assignment

对每一代 detector 都输出类似：

```text
GT box          = 红框
candidate point = 灰点
positive        = 绿点
selected anchor = 蓝框
```

理解 YOLO 最容易卡住的不是 Conv，而是：

```text
GT 到底监督了谁？
```

---

# 13. 推荐的源码阅读顺序

不要从 `train.py` 第一行一路往下读。

建议按数据流：

```text
1. model YAML
   ↓
2. module definitions
   ↓
3. forward()
   ↓
4. Detect Head raw output
   ↓
5. bbox decode
   ↓
6. target / assigner
   ↓
7. loss
   ↓
8. postprocess / NMS
   ↓
9. trainer
   ↓
10. augmentation / optimizer / EMA 等工程组件
```

每读一个模块都问：

```text
输入 shape 是什么？
输出 shape 是什么？
训练和推理是否一样？
它有没有参数？
有没有状态？
梯度流向哪里？
上一代为什么没有它？
下一代为什么又改掉它？
```

---

# 14. 推荐的总学习主线

最终不要按“版本号”记忆，而应建立下面几条技术主线。

## 主线 A：Backbone

```text
Bottleneck
   ↓
C3 / CSP
   ↓
C2f
   ↓
C3k2
   ↓
C2PSA / attention integration
```

## 主线 B：Feature Pyramid

```text
FPN
 ↓
FPN + PAN
 ↓
同一框架下持续改进 fusion block
```

## 主线 C：Prediction Head

```text
Anchor-based Coupled
        ↓
Anchor-free Decoupled
        ↓
Dual O2M/O2O End-to-End
```

## 主线 D：BBox

```text
Anchor offset
    ↓
Point + LTRB
    ↓
LTRB distribution + DFL
    ↓
Direct distance regression / DFL-free
```

## 主线 E：Assignment

```text
Anchor matching
    ↓
Anchor ratio + neighbor grid
    ↓
TaskAlignedAssigner
    ↓
O2M/O2O + STAL
```

## 主线 F：Post-processing

```text
Dense prediction
    ↓
NMS
    ↓
One-to-One
    ↓
NMS-free
```

---

# 15. 最终希望达到的能力

完成整条复现路线后，目标不是“自己写了五个 YOLO”，而是应该能回答：

- [ ] 为什么 YOLO 要做多尺度检测？
- [ ] FPN 和 PAN 分别在传什么信息？
- [ ] CSP/C3/C2f/C3k2 为什么不断变化？
- [ ] Anchor 到底解决了什么，又制造了什么问题？
- [ ] Anchor-free 为什么不代表“没有 anchor point”？
- [ ] Objectness 为什么后来可以删除？
- [ ] Coupled Head 为什么会有任务冲突？
- [ ] TaskAlignedAssigner 比静态 anchor matching 强在哪里？
- [ ] DFL 为什么可以提高定位表达能力？
- [ ] YOLO26 为什么又删除 DFL？
- [ ] One-to-Many 为什么天然产生重复框？
- [ ] NMS 实际是在修正训练目标的什么特性？
- [ ] One-to-One 为什么可以让 NMS 消失？
- [ ] 为什么 One-to-One 又会带来正样本不足的问题？
- [ ] STAL 在解决什么？
- [ ] Progressive Loss 为什么适合 Dual Head？
- [ ] 网络结构收益、训练 recipe 收益、label assignment 收益应该怎样通过 ablation 区分？

如果这些问题都能从代码、公式和实验结果三个层面回答，就基本从“会使用 YOLO”进入了“理解 YOLO 检测器设计”的阶段。

---

# 16. 后续文档拆分建议

随着复现深入，可以把本总纲拆成：

```text
00_YOLO_evolution_overview.md

01_YOLOv3/
├── 01_darknet53.md
├── 02_fpn.md
├── 03_anchor_and_grid.md
├── 04_bbox_decode.md
├── 05_target_assignment.md
├── 06_loss.md
└── 07_nms.md

02_YOLOv5/
├── 01_csp_c3.md
├── 02_pan.md
├── 03_sppf.md
├── 04_bbox_parameterization.md
├── 05_neighbor_assignment.md
└── 06_training_recipe.md

03_YOLOv8/
├── 01_c2f.md
├── 02_decoupled_head.md
├── 03_anchor_free.md
├── 04_objectness_free.md
├── 05_task_aligned_assigner.md
├── 06_dfl.md
└── 07_loss.md

04_YOLO11/
├── 01_c3k2.md
├── 02_c2psa.md
└── 03_lightweight_head.md

05_YOLO26/
├── 01_sppf_residual.md
├── 02_dfl_free.md
├── 03_distance_l1.md
├── 04_dual_head.md
├── 05_one_to_one.md
├── 06_nms_free.md
├── 07_stal.md
├── 08_progressive_loss.md
└── 09_musgd.md
```

这会比把所有知识堆在一个巨大的 Markdown 文件中更适合长期维护。

---

# 17. 官方资料索引

以下资料应作为后续源码复现时的主要对照，而不是第三方二手文章。

1. **Ultralytics — YOLO Architecture Explained: From YOLOv3 to YOLO26**  
   https://docs.ultralytics.com/guides/yolo-architecture/

2. **Ultralytics — YOLOv3 and YOLOv3u**  
   https://docs.ultralytics.com/models/yolov3/

3. **Ultralytics — YOLOv5 Architecture**  
   https://docs.ultralytics.com/yolov5/tutorials/architecture-description/

4. **Ultralytics — YOLO11**  
   https://docs.ultralytics.com/models/yolo11/

5. **Ultralytics — YOLO26**  
   https://docs.ultralytics.com/models/yolo26/

6. **Ultralytics — YOLO26 Training Recipe**  
   https://docs.ultralytics.com/guides/yolo26-training-recipe/

7. **Ultralytics GitHub**  
   https://github.com/ultralytics/ultralytics

8. **Ultralytics YOLOv5 GitHub**  
   https://github.com/ultralytics/yolov5

9. **Ultralytics YOLOv3 GitHub**  
   https://github.com/ultralytics/yolov3

---

# 18. 当前路线的第一站

下一阶段建议从：

```text
YOLOv3：最小可训练实现
```

开始，而不是直接复刻完整官方工程。

第一版只保留：

```text
Darknet-53
FPN
3-scale anchor head
basic target assignment
CIoU
BCE objectness
BCE classification
NMS
```

暂时不要加入：

```text
Mosaic
MixUp
EMA
AutoAnchor
复杂 scheduler
AMP
DDP
多尺度训练
大量工程优化
```

先把：

```text
forward
→ assignment
→ loss
→ backward
→ decode
→ NMS
```

这一条完整链路彻底跑通，再逐步进入 YOLOv5。

---

> 文档定位：这是整个复现项目的总纲。后续每完成一个版本/组件，都应把实验结果、源码差异、公式、tensor shape 和结论回填到对应章节或拆分后的子文档中。
