"""YOLOv3 / Darknet data loading without random augmentation."""

from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Load an image directory or a Darknet image-list text file.

    Args:
        image_source: Image directory (searched recursively), or a text file
            containing one image path per line. Relative entries are resolved
            against the list file's directory, unless ``root`` is supplied.
        img_size: Square input size; must be a positive multiple of 32.
        num_classes: Optional class count, for validating zero-based class IDs.
        max_boxes: Number of truth rows, defaulting to Darknet YOLOv3's 90.
        root: Base directory for relative paths inside an image-list file.

    Returns:
        image: float32 RGB tensor of shape (3, img_size, img_size), in [0, 1].
        target: float32 tensor of shape (max_boxes, 5), with rows ordered as
            (x_center, y_center, width, height, class_id). Coordinates are
            normalized to the letterboxed image. Unused rows are all zeros.

    Label files contain ``class_id x_center y_center width height``, normalized
    to the original image. An images/JPEGImages/raw directory component is
    replaced with labels; otherwise the .txt file is next to the image.
    Empty label files represent background images; missing files are errors.
    """

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, image_source, img_size=416, num_classes=None, max_boxes=90, root=None):
        if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size <= 0 or img_size % 32:
            raise ValueError("img_size must be a positive integer multiple of 32")
        if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes <= 0:
            raise ValueError("max_boxes must be a positive integer")
        if num_classes is not None and (
            isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer or None")
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        source = Path(image_source).expanduser().resolve()
        if source.is_dir():
            self.image_paths = sorted(
                p for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES
            )
        else:
            raise FileNotFoundError(f"Expected an image directory or image-list .txt file: {source}")
        if not self.image_paths:
            raise ValueError(f"No images found in {source}")
        self.label_paths = [self._label_path(p) for p in self.image_paths]

    @staticmethod
    def _label_path(image_path):
        parts = list(image_path.parts)
        for index in range(len(parts) - 2, -1, -1):
            if parts[index] in {"images", "JPEGImages", "raw"}:
                parts[index] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def __len__(self):
        return len(self.image_paths)

    def _read_labels(self, label_path):
        rows = label_path.read_text(encoding="utf-8-sig").splitlines()
        rows = [row.split() for row in rows if row.strip() != '']
        return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        labels = self._read_labels(self.label_paths[index])
        with Image.open(image_path) as image:
            # Keep encoded orientation, as Darknet does; do not rotate by EXIF.
            pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width = pixels.shape[:2]
        size = self.img_size
        if width >= height:
            new_w, new_h = size, max(1, height * size // width)
        else:
            new_w, new_h = max(1, width * size // height), size
        left, top = (size - new_w) // 2, (size - new_h) // 2    # 灰边填充宽度
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        # Darknet resize_image aligns the endpoints of each axis. In contrast,
        # PIL/OpenCV's usual bilinear resize uses a different sampling grid.
        resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=True)[0]
        boxed = torch.full((3, size, size), 0.5, dtype=torch.float32)
        boxed[:, top:top + new_h, left:left + new_w] = resized

        if len(labels) > self.max_boxes:
            warnings.warn(
                f"{image_path}: {len(labels)} boxes exceed max_boxes={self.max_boxes}; "
                "keeping the first rows in file order",
                UserWarning, stacklevel=2,
            )
        # Darknet caps the number of rows before removing tiny boxes.
        labels = labels[:self.max_boxes]
        target = np.zeros((self.max_boxes, 5), dtype=np.float32)
        if len(labels):
            centers, sizes = labels[:, 1:3], labels[:, 3:5]
            scale = np.array([new_w, new_h], dtype=np.float32) / size
            offset = np.array([left, top], dtype=np.float32) / size
            low = np.clip((centers - sizes / 2) * scale + offset, 0, 1)
            high = np.clip((centers + sizes / 2) * scale + offset, 0, 1)
            centers = (low + high) / 2
            sizes = high - low
            valid = (sizes >= 0.001).all(axis=1)
            truth = np.concatenate((centers, sizes, labels[:, :1]), axis=1)[valid]
            target[:len(truth)] = truth
        return boxed, torch.from_numpy(target)
