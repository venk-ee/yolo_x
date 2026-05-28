import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.ops import generalized_box_iou

def cxcywh_to_xyxy(box):
    if box.dim()==1:
        box=box.unsqueeze(0)
    cx,cy,w,h=box.split(1,dim=-1)
    x1=cx-(w/2)
    y1=cy-(h/2)
    x2=cx+(w/2)
    y2=cy+(h/2)
    return torch.cat([x1,y1,x2,y2],dim=-1)


def xyxy_to_cxcywh(box):
    if box.dim()==1:
        box=box.unsqueeze(0)
    x1,y1,x2,y2=box.split(1,dim=-1)
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1
    return torch.cat([cx,cy,w,h],dim=-1)

def get_box_IoU(box1,box2):
    box1=cxcywh_to_xyxy(box1)
    box2=cxcywh_to_xyxy(box2)
    return box_iou(box1,box2)

def  get_dynamic_k(ious):
    # ious is a tensor of shape [num_preds]
    if ious.size(0) ==0:
        return 1
    topk_num = min(10, ious.size(0))
    topk_ious=torch.sum(torch.topk(ious, k=topk_num).values)
    return max(1, int(topk_ious))

def simota_matcher(pred_boxes, pred_cls, gt_boxes, gt_cls):
    num_gt = gt_boxes.size(0)
    num_preds = pred_boxes.size(0)

    if num_gt == 0 or num_preds == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    ious = get_box_IoU(gt_boxes, pred_boxes)

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

    matching_matrix=torch.zeros((num_gt,num_preds),dtype=torch.bool)

    for gt_idx in range(num_gt):
        
        gt_iou = ious[gt_idx]
        
        dynamic_k = get_dynamic_k(gt_iou)
        
        gt_cost = cost_matrix[gt_idx]

        topk_index=torch.topk(gt_cost,k=dynamic_k,largest=False).indices

        matching_matrix[gt_idx,topk_index]=True
        
    claims_per_pred = matching_matrix.sum(dim=0)

    if (claims_per_pred > 1).any():

        cost_matrix_masked = cost_matrix.clone()
        cost_matrix_masked[~matching_matrix] = float('inf')

        best_gt_for_each_pred = torch.argmin(cost_matrix_masked, dim=0)
        

        matching_matrix.zero_()

        valid_preds = claims_per_pred > 0

        matching_matrix[best_gt_for_each_pred[valid_preds], valid_preds] = True

    final_gts, final_preds = torch.where(matching_matrix)

    return final_gts, final_preds