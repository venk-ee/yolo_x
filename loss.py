import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from utils import cxcywh_to_xyxy, simota_matcher

##--NOTE:AI WAS USED IN DEVLOPMENT OF THIS SCRIPT-------

class YOLOXLoss(nn.Module):
    def __init__(self,num_classes=80, strides=[8, 16, 32]):
        super().__init__()

        self.strides=strides
        self.num_classes = num_classes

    def get_grid_points(self, H, W, stride, device):
        # We pass H and W directly now!
        xs = (torch.arange(W, device=device) + 0.5) * stride
        ys = (torch.arange(H, device=device) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        grid_points = torch.stack([grid_x, grid_y], dim=-1)
        return grid_points


    def decode_boxes(self,reg_pred,grid_points,stride):
        # reg_pred is [N, 4] -> dx, dy, dw, dh
        # grid_points is [N, 2] -> grid_x, grid_y

        # 1. Add grid offsets and multiply by stride for x and y
        cx = (reg_pred[:, 0] + grid_points[:, 0]) * stride
        cy = (reg_pred[:, 1] + grid_points[:, 1]) * stride
        
        # 2. Exponentiate width and height and multiply by stride
        w = torch.exp(reg_pred[:, 2]) * stride
        h = torch.exp(reg_pred[:, 3]) * stride

        # 3. Stack them back together!
        decoded_boxes = torch.stack([cx, cy, w, h], dim=-1)
        return decoded_boxes
    
    def iou_loss(self, pred_boxes, gt_boxes):
        # 1. Convert both to xyxy
        pred_xyxy = cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = cxcywh_to_xyxy(gt_boxes)
        
        # 2. Calculate GIoU. It gives an [N, N] matrix.
        giou_matrix = generalized_box_iou(pred_xyxy, gt_xyxy)
        
        # 3. We only care about the direct matches (the diagonal!)
        giou_scores = giou_matrix.diag()
        
        # 4. Loss is 1 - GIoU
        loss = 1.0 - giou_scores
        return loss.sum() # Return the sum of the loss

    def forward(self, predictions, gt_boxes_list, gt_cls_list):
        """
        predictions:   list of 3 tuples [(cls_out, reg_out, obj_out), ...]
                       each cls_out: [B, num_classes, H, W]
                       each reg_out: [B, 4, H, W]
                       each obj_out: [B, 1, H, W]
        gt_boxes_list: list of B tensors, each [num_gt, 4] cxcywh
        gt_cls_list:   list of B tensors, each [num_gt] long
        """
        device = predictions[0][0].device
        B = predictions[0][0].size(0)

        # Step 1: flatten each scale, collect grid points
        all_cls, all_reg, all_obj, all_grids = [], [], [], []

        for (cls_out, reg_out, obj_out), stride in zip(predictions, self.strides):
            B, C, H, W = cls_out.shape
            # Flatten H and W, then permute the channels to the end
            cls_out = cls_out.view(B, self.num_classes, -1).permute(0, 2, 1)
            reg_out = reg_out.view(B, 4, -1).permute(0, 2, 1)
            obj_out = obj_out.view(B, 1, -1).permute(0, 2, 1)
            grids = self.get_grid_points(H, W, stride, device)  # [H*W, 2]
            all_cls.append(cls_out)
            all_reg.append(reg_out)
            all_obj.append(obj_out)
            all_grids.append(grids)

        # [B, total_preds, ...]
        all_cls  = torch.cat(all_cls,  dim=1)
        all_reg  = torch.cat(all_reg,  dim=1)
        all_obj  = torch.cat(all_obj,  dim=1)
        all_grids = torch.cat(all_grids, dim=0)  # [total_preds, 2]

        # Step 2: loop over batch
        total_cls_loss = torch.tensor(0., device=device)
        total_reg_loss = torch.tensor(0., device=device)
        total_obj_loss = torch.tensor(0., device=device)
        num_positives  = 0

        for i in range(B):
            cls_i  = all_cls[i]   # [total_preds, num_classes]
            reg_i  = all_reg[i]   # [total_preds, 4]
            obj_i  = all_obj[i]   # [total_preds, 1]
            gt_b   = gt_boxes_list[i]
            gt_c   = gt_cls_list[i]

            # 1. Decode reg_i using all_grids
            # Note: stride is 1 because the grids already account for it!
            decoded_boxes = self.decode_boxes(reg_i, all_grids, stride=1)

            # 2. Run SimOTA
            gt_idx, pred_idx = simota_matcher(decoded_boxes, cls_i, gt_b, gt_c)

            # 3. Objectness targets — all zeros, then set positives to 1
            obj_targets = torch.zeros_like(obj_i)
            if pred_idx.numel() > 0:
                obj_targets[pred_idx] = 1.0
            total_obj_loss += F.binary_cross_entropy_with_logits(obj_i, obj_targets, reduction="sum")

            # 4. If no positives, skip cls/reg
            if pred_idx.numel() == 0: 
                continue

            # 5. Cls loss on positives only
            gt_classes_for_winners = gt_c[gt_idx]
            cls_targets = F.one_hot(gt_classes_for_winners, num_classes=self.num_classes).float()
            total_cls_loss += F.binary_cross_entropy_with_logits(
                cls_i[pred_idx], cls_targets, reduction="sum"
            )

            # 6. Reg loss on positives only
            total_reg_loss += self.iou_loss(decoded_boxes[pred_idx], gt_b[gt_idx])

            num_positives += pred_idx.numel()


        # Normalise by number of positives (not batch size)
        norm = max(num_positives, 1)
        return (total_cls_loss + total_reg_loss + total_obj_loss) / norm