import torch
import torchvision
from pycocotools.coco import COCO

import albumentations as A
import os
import cv2


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

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                category_ids=category_ids,
            )

            image = transformed["image"]
            boxes = transformed["bboxes"]
            category_ids = transformed["category_ids"]

        resized_h, resized_w = image.shape[:2]

        new_boxes = []

        for box in boxes:
            cx = box[0] + (box[2] / 2.0)
            cy = box[1] + (box[3] / 2.0)
            w = box[2]
            h = box[3]

            new_boxes.append([cx, cy, w, h])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        # keypoints = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, 2, 3)

        # args={}

        # args['bbox']=torch.tensor(new_boxes,dtype=torch.float32)
        # args['image_id']=torch.tensor([t['image_id']for t in target],dtype=torch.int64)
        # args['iscrowd']=torch.tensor([t['iscrowd'] for t in target], dtype=torch.uint8),
        # args['area']=((boxes[:,3]-boxes[:,1]) *(boxes[:,2]- boxes[:,0]) )
        # args['labels']=torch.tensor([t['category_id']for t in target],dtype= torch.int64)
        # args['keypoints']=keypoints

        args = {
            "orig_size": torch.tensor([orig_w, orig_h]),
            "resized_size": torch.tensor([resized_w, resized_h]),
            "boxes": boxes,
            "image_id": torch.tensor(
                [t["image_id"] for t in target], dtype=torch.int64
            ),
            "iscrowd": torch.tensor([t["iscrowd"] for t in target], dtype=torch.uint8),
            "area": boxes[:, 2] * boxes[:, 3],
            "labels": torch.tensor(
                [t["category_id"] for t in target], dtype=torch.int64
            ),
            # "keypoints": keypoints,
        }

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image / 255.0, args

    def __len__(self):
        return len(self.ids)


def get_transform(train: bool):
    if train:
        return A.Compose(
            [A.Resize(640, 640)],
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
        )
    else:
        return A.Compose(
            [A.Resize(640, 640)],
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
        )
