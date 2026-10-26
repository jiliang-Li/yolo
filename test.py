import torch
from yolov3 import *


data = torch.randn(1, 3, 416, 416)
backbone = Darknet53()
neck = Yolov3Neck()
head = Yolov3Head(n_classes=15)
c5, c4, c3 = backbone(data)
p5, p4, p3 = neck((c5, c4, c3))
f5, f4, f3 = head((p5, p4, p3))
print(c5.shape, c4.shape, c3.shape)
print(p5.shape, p4.shape, p3.shape)
print(f5.shape, f4.shape, f3.shape)