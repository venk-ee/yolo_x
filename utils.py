import torch
import torchvision
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.ops import generalized_box_iou
import math
import copy
from torch.utils.data import DataLoader

from data import coco_data, get_transform

from pycocotools.cocoeval import COCOeval


def cxcywh_to_xyxy(box):
    if box.dim() == 1:
        box = box.unsqueeze(0)
    cx, cy, w, h = box.split(1, dim=-1)
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    return torch.cat([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(box):
    if box.dim() == 1:
        box = box.unsqueeze(0)
    x1, y1, x2, y2 = box.split(1, dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.cat([cx, cy, w, h], dim=-1)


def get_box_IoU(box1, box2):
    box1 = cxcywh_to_xyxy(box1)
    box2 = cxcywh_to_xyxy(box2)
    return box_iou(box1, box2)


def get_dynamic_k(ious):
    # ious is a tensor of shape [num_preds]
    if ious.size(0) == 0:
        return 1
    ious = torch.nan_to_num(ious, nan=0.0, posinf=0.0, neginf=0.0)
    topk_num = min(10, ious.size(0))
    topk_ious = torch.sum(torch.topk(ious, k=topk_num).values)
    return max(1, int(topk_ious))


def simota_matcher(
    pred_boxes,
    pred_cls,
    gt_boxes,
    gt_cls,
    anchor_centers=None,
    strides=None,
    center_radius=2.5,
):
    num_gt = gt_boxes.size(0)
    num_preds = pred_boxes.size(0)

    if num_gt == 0 or num_preds == 0:
        return torch.zeros(0, dtype=torch.long, device=pred_boxes.device), torch.zeros(
            0, dtype=torch.long, device=pred_boxes.device
        )

    ious = get_box_IoU(gt_boxes, pred_boxes)
    ious = torch.nan_to_num(ious, nan=0.0, posinf=0.0, neginf=0.0)

    loss_reg = -torch.log(ious + 1e-8)

    num_classes = pred_cls.size(-1)
    gt_cls_one_hot = F.one_hot(gt_cls, num_classes).float()

    pred_cls_exp = pred_cls.unsqueeze(0).expand(num_gt, -1, -1)

    # Soft IoU-weighted targets
    iou_weights = ious.unsqueeze(-1)
    target_cls = gt_cls_one_hot.unsqueeze(1).expand(-1, num_preds, -1) * iou_weights

    loss_cls = F.binary_cross_entropy_with_logits(
        pred_cls_exp, target_cls, reduction="none"
    ).sum(-1)

    cost_matrix = loss_cls + 3.0 * loss_reg

    if anchor_centers is not None and strides is not None:
        gt_centers = gt_boxes[:, :2]
        strides = strides.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
        radius = center_radius * strides.unsqueeze(0)
        center_dist_x = (
            anchor_centers[:, 0].unsqueeze(0) - gt_centers[:, 0].unsqueeze(1)
        ).abs()
        center_dist_y = (
            anchor_centers[:, 1].unsqueeze(0) - gt_centers[:, 1].unsqueeze(1)
        ).abs()
        candidate_mask = (center_dist_x < radius) & (center_dist_y < radius)
    else:
        candidate_mask = torch.ones(
            (num_gt, num_preds), dtype=torch.bool, device=pred_boxes.device
        )

    matching_matrix = torch.zeros(
        (num_gt, num_preds), dtype=torch.bool, device=pred_boxes.device
    )

    for gt_idx in range(num_gt):
        gt_candidate_mask = candidate_mask[gt_idx]
        if gt_candidate_mask.any():
            candidate_indices = torch.nonzero(
                gt_candidate_mask, as_tuple=False
            ).squeeze(1)
            gt_iou = ious[gt_idx, candidate_indices]
            gt_cost = cost_matrix[gt_idx, candidate_indices]
        else:
            candidate_indices = torch.arange(num_preds, device=pred_boxes.device)
            gt_iou = ious[gt_idx]
            gt_cost = cost_matrix[gt_idx]

        dynamic_k = get_dynamic_k(gt_iou)
        topk_local = torch.topk(
            gt_cost, k=min(dynamic_k, gt_cost.numel()), largest=False
        ).indices
        topk_index = candidate_indices[topk_local]

        matching_matrix[gt_idx, topk_index] = True

    claims_per_pred = matching_matrix.sum(dim=0)

    if (claims_per_pred > 1).any():
        cost_matrix_masked = cost_matrix.clone()
        cost_matrix_masked[~matching_matrix] = float("inf")

        best_gt_for_each_pred = torch.argmin(cost_matrix_masked, dim=0)

        matching_matrix.zero_()

        valid_preds = claims_per_pred > 0

        matching_matrix[best_gt_for_each_pred[valid_preds], valid_preds] = True

    final_gts, final_preds = torch.where(matching_matrix)

    return final_gts, final_preds


class ModelEMA:
    def __init__(self, model, device, decay=0.9999, updates=0):
        self.ema = copy.deepcopy(model).eval().to(device)
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.ema.state_dict().values(), model.state_dict().values()
            ):
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * d + (1.0 - d) * model_v.detach())


def get_optimizer(model, lr=0.001, weight_decay=5e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, epochs, warmup_epochs=5, warmup_lr=0.01, min_lr=1e-5):
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_lr, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
            ),
        ],
        milestones=[warmup_epochs],
    )


def get_devices() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images)
    return images, targets


def get_data_loader(
    train=True,
    val=True,
    test=False,
    BATCH_SIZE=32,
    train_transform=get_transform(True),
    val_transform=get_transform(False),
    test_transform=get_transform(False),
):
    train_data_loader = None
    val_data_loader = None
    test_data_loader = None
    if train:
        train_dataset = coco_data(
            root="/mnt/ken/exe_real_coco",
            split="",
            anno="annotations/instances_train.json",
            transforms=train_transform,
            image_folder="images/train",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
        train_data_loader = train_loader
    if test:
        test_dataset = coco_data(
            root="/mnt/ken/exe_real_coco",
            split="",
            anno="annotations/instances_test.json",
            transforms=test_transform,
            image_folder="images/test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
        test_data_loader = test_loader

    if val:
        val_dataset = coco_data(
            root="/mnt/ken/exe_real_coco",
            split="",
            anno="annotations/instances_val.json",
            transforms=val_transform,
            image_folder="images/val",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
        val_data_loader = val_loader

    return train_data_loader, val_data_loader, test_data_loader


def get_grid_points(H, W, stride, device):
    # We pass H and W directly now!
    xs = torch.arange(W, device=device) + 0.5
    ys = torch.arange(H, device=device) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    grid_points = torch.stack([grid_x, grid_y], dim=-1)
    return grid_points


def decode_boxes(reg_pred, grid_points, stride=1):
    """Decode raw regression outputs into [cx, cy, w, h] coordinates."""
    cx = (reg_pred[:, 0] + grid_points[:, 0]) * stride
    cy = (reg_pred[:, 1] + grid_points[:, 1]) * stride
    w = torch.exp(reg_pred[:, 2].clamp(max=8.0)) * stride
    h = torch.exp(reg_pred[:, 3].clamp(max=8.0)) * stride
    return torch.stack([cx, cy, w, h], dim=-1)


def post_process_nms(outputs, strides=[8, 16, 32], score_thresh=0.05, iou_thresh=0.5):
    """
    Takes raw model outputs and returns filtered detections after NMS.

    Args:
        outputs: tuple of 3 tuples -> (shallow, mid, deep)
                 each is (cls_out, reg_out, obj_out)
        strides: the stride for each head
        score_thresh: minimum confidence to keep a box
        iou_thresh: IoU threshold for NMS overlap removal

    Returns:
        list of (boxes, scores, labels) per image in the batch
        boxes are in [xmin, ymin, xmax, ymax] format
    """
    device = outputs[0][0].device
    B = outputs[0][0].size(0)

    all_cls, all_reg, all_obj, all_grids, all_strides = [], [], [], [], []

    # Step 1: Flatten all 3 heads (same logic as loss.py lines 74-84)
    for (cls_out, reg_out, obj_out), stride in zip(outputs, strides):
        _, C, H, W = cls_out.shape
        all_cls.append(cls_out.view(B, C, -1).permute(0, 2, 1))
        all_reg.append(reg_out.view(B, 4, -1).permute(0, 2, 1))
        all_obj.append(obj_out.view(B, 1, -1).permute(0, 2, 1))
        grids = get_grid_points(H, W, stride, device)
        all_grids.append(grids)
        all_strides.append(torch.full_like(grids[:, 0], stride))

    # Concatenate across all scales
    all_cls = torch.cat(all_cls, dim=1).sigmoid()  # [B, N, num_classes]
    all_reg = torch.cat(all_reg, dim=1)  # [B, N, 4]
    all_obj = torch.cat(all_obj, dim=1).sigmoid()  # [B, N, 1]
    all_grids = torch.cat(all_grids, dim=0)  # [N, 2]
    all_strides = torch.cat(all_strides, dim=0)  # [N]

    # Step 2: Process each image in the batch
    results = []
    for i in range(B):
        # Decode raw regression into [cx, cy, w, h]
        decoded_cxcywh = decode_boxes(all_reg[i], all_grids, all_strides)

        # Convert to [xmin, ymin, xmax, ymax] for NMS
        decoded_xyxy = cxcywh_to_xyxy(decoded_cxcywh)

        # Final score = objectness * class_probability
        scores_per_class = all_obj[i] * all_cls[i]  # [N, num_classes]

        # Find the best class and its score for each box
        max_scores, class_ids = scores_per_class.max(dim=1)

        # Filter out low-confidence boxes
        mask = max_scores > score_thresh
        boxes = decoded_xyxy[mask]
        scores = max_scores[mask]
        labels = class_ids[mask]

        # Run NMS to remove overlapping boxes
        keep = torchvision.ops.nms(boxes, scores, iou_thresh)

        results.append((boxes[keep], scores[keep], labels[keep]))

    return results


def eval_metrics_coco_bbox(coco_gt, predictions_list):
    if len(predictions_list) == 0:
        return 0.0
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}

    # Load the predictions into a COCO object
    coco_dt = coco_gt.loadRes(predictions_list)

    # Initialize the COCO evaluator for bounding boxes
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Run the math
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats[0] contains the mAP score across IoU thresholds 0.50:0.95!
    return coco_eval.stats[0]


def format_for_coco(results, targets):
    batch_preds = []

    # Loop over each image in the batch
    for i in range(len(results)):
        boxes, scores, labels = results[i]
        image_id = targets[i]["image_id"][0].item()
        orig_w, orig_h = targets[i]["orig_size"].tolist()
        resized_w, resized_h = targets[i]["resized_size"].tolist()
        label_to_cat_id = targets[i].get("label_to_cat_id")

        scale_x = orig_w / float(resized_w)
        scale_y = orig_h / float(resized_h)

        # Loop over every surviving box for this image
        for j in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[j].tolist()

            xmin = xmin * scale_x
            xmax = xmax * scale_x
            ymin = ymin * scale_y
            ymax = ymax * scale_y

            w = xmax - xmin
            h = ymax - ymin
            label_idx = int(labels[j].item())

            batch_preds.append(
                {
                    "image_id": image_id,
                    "category_id": (
                        label_to_cat_id[label_idx].item()
                        if label_to_cat_id is not None
                        else label_idx
                    ),
                    "bbox": [xmin, ymin, w, h],
                    "score": scores[j].item(),
                }
            )

    return batch_preds
