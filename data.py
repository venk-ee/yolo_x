import torch
import torchvision
from pycocotools.coco import COCO

import albumentations as A
import os
import cv2


IMAGE_SIZE = 640
LETTERBOX_COLOR = (114, 114, 114)


def letterbox_image(image, new_shape=IMAGE_SIZE, color=LETTERBOX_COLOR):
    """Resize while keeping aspect ratio, then pad to a square."""
    orig_h, orig_w = image.shape[:2]
    scale = min(new_shape / float(orig_h), new_shape / float(orig_w))

    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))

    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - resized_w
    pad_h = new_shape - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    letterboxed = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return letterboxed, scale, pad_left, pad_top, resized_w, resized_h


def letterbox_boxes_coco(boxes, scale, pad_left, pad_top):
    letterboxed_boxes = []
    for box in boxes:
        x, y, w, h = box
        letterboxed_boxes.append(
            [x * scale + pad_left, y * scale + pad_top, w * scale, h * scale]
        )
    return letterboxed_boxes


def unletterbox_boxes_xyxy(boxes, scale, pad_left, pad_top, orig_w, orig_h):
    if boxes.numel() == 0:
        return boxes

    boxes = boxes.clone()
    boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

    boxes[:, 0].clamp_(0, orig_w)
    boxes[:, 2].clamp_(0, orig_w)
    boxes[:, 1].clamp_(0, orig_h)
    boxes[:, 3].clamp_(0, orig_h)
    return boxes


class coco_data(torchvision.datasets.VisionDataset):
    def __init__(
        self, root=None, split=None, anno=None, transforms=None, image_folder=None
    ):
        super().__init__(
            root,
            transforms,
        )

        self.root = root
        self.split = split
        self.anno = anno
        self.image_folder = image_folder
        self.coco = COCO(os.path.join(self.root, self.split, self.anno))
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label_to_cat_id = torch.tensor(self.cat_ids, dtype=torch.int64)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if len(self.get_target(id)) > 0]

    def get_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, self.split, self.image_folder, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self.get_image(id)
        orig_h, orig_w = image.shape[:2]
        target = self.get_target(id)
        boxes = [t["bbox"] for t in target]
        category_ids = [t["category_id"] for t in target]
        iscrowd = [t["iscrowd"] for t in target]

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                category_ids=category_ids,
                iscrowd=iscrowd,
            )

            image = transformed["image"]
            boxes = transformed["bboxes"]
            category_ids = transformed["category_ids"]
            iscrowd = transformed["iscrowd"]

        image, scale, pad_left, pad_top, resized_w, resized_h = letterbox_image(image)
        boxes = letterbox_boxes_coco(boxes, scale, pad_left, pad_top)

        new_boxes = []

        for box in boxes:
            cx = box[0] + (box[2] / 2.0)
            cy = box[1] + (box[3] / 2.0)
            w = box[2]
            h = box[3]

            new_boxes.append([cx, cy, w, h])

        if len(new_boxes) > 0:
            boxes = torch.tensor(new_boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        args = {
            "orig_size": torch.tensor([orig_w, orig_h]),
            "resized_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
            "letterbox_scale": torch.tensor(scale, dtype=torch.float32),
            "letterbox_pad": torch.tensor([pad_left, pad_top], dtype=torch.float32),
            "boxes": boxes,
            "image_id": torch.tensor(
                [t["image_id"] for t in target], dtype=torch.int64
            ),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.uint8),
            "area": boxes[:, 2] * boxes[:, 3],
            "labels": torch.tensor(
                [self.cat_id_to_label[cat_id] for cat_id in category_ids],
                dtype=torch.int64,
            ),
            "label_to_cat_id": self.label_to_cat_id,
            # "keypoints": keypoints,
        }

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image / 255.0, args

    def __len__(self):
        return len(self.ids)


def get_transform(train: bool):
    # Note: Bounding boxes and labels are automatically transformed/rotated 
    # by Albumentations to match the modified image using the bbox_params settings.
    if train:
        return A.Compose(
            [
                # Geometric transforms (YOLO standard style: horizontal flip, scale & translate)
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.5, 1.5),
                    translate_percent=(-0.1, 0.1),
                    rotate=0,
                    shear=0,
                    p=0.5,
                ),
                # Color transforms (YOLO HSV augmentation)
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_ids", "iscrowd"],
                min_area=5,
                min_visibility=0.1,
            ),
        )
    else:
        # Validation/testing requires clean images without augmentations
        return A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_ids", "iscrowd"],
                min_area=1,
                min_visibility=0.01,
            ),
        )

