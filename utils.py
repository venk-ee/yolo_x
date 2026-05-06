import torch
import torch.nn.functional as F
import torchvision

def decode_box(reg_pred, stride=16):
    
    batch_size,_,h,w=reg_pred.shape

    y_grid,x_grid=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')

    y_grid=y_grid.unsqueeze(0)
    x_grid=x_grid.unsqueeze(0)
    dx = reg_pred[:, 0, :, :]
    dy = reg_pred[:, 1, :, :]
    dw = reg_pred[:, 2, :, :]
    dh = reg_pred[:, 3, :, :]

    center_x = (x_grid + dx) * stride
    center_y = (y_grid + dy) * stride
    width = torch.exp(dw) * stride
    height = torch.exp(dh) * stride

    decoded_boxes = torch.stack([center_x, center_y, width, height], dim=1)
    return decoded_boxes


def cxcywh2xyxy(decoded_boxes):
    center_x, center_y, width, height = decoded_boxes.split(1, dim=-1)

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    return torch.cat([x1, y1, x2, y2], dim=-1)   

def prepare_for_nms(decoded_boxes,obj_pred,cls_preds):

    batch_size=decoded_boxes.shape[0]
    flat_boxes=decoded_boxes.permute(0,2,3,1)
    flat_boxes=flat_boxes.reshape(batch_size,-1,4)

    flat_obj=obj_pred.permute(0,2,3,1)
    flat_obj=flat_obj.reshape(batch_size,-1,1)

    flat_cls=cls_preds.permute(0,2,3,1)
    flat_cls=flat_cls.reshape(batch_size,-1,80)
    
    return flat_boxes,flat_obj,flat_cls


def post_processing(flat_boxes,flat_obj,flat_cls,conf_thres=0.05,nms_thres=0.5):

    boxes = flat_boxes[0]
    obj_scores = flat_obj[0]
    class_preds = flat_cls[0]

    valid_mask = (obj_scores > conf_thres).squeeze()
    
    boxes = boxes[valid_mask]
    obj_scores = obj_scores[valid_mask]
    class_preds = class_preds[valid_mask]
    
    if boxes.shape[0] == 0:
        return None

    class_scores, class_ids = torch.max(class_preds, dim=1, keepdim=True)

    final_scores = obj_scores * class_scores


    boxes_xyxy = cxcywh2xyxy(boxes)
    # boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

    keep_indices = torchvision.ops.nms(boxes_xyxy, final_scores.squeeze(), nms_thres)

    final_boxes = boxes_xyxy[keep_indices]
    final_scores = final_scores[keep_indices]
    final_class_ids = class_ids[keep_indices]

    detections = torch.cat([final_boxes, final_scores, final_class_ids.float()], dim=1)

    return detections

########################### SimOTA  ###########################################

def calculate_loss_matrix(class_pred,box_preds,gt_class,gt_box):
    # cls_preds: [25, 80] -> The 80 class scores for the 25 surviving candidates
    # box_preds: [25, 4]  -> The xyxy boxes for the 25 surviving candidates
    # gt_class:  Integer  -> The correct class ID (e.g., 16 for Dog)
    # gt_box:    [1, 4]   -> The real, ground-truth xyxy box of the dog

    # classificartion loss
    #create a answer key (zeros everywhere except a 1.0 at the correct class ID)
    ans_key=torch.zeros_like(class_pred)
    ans_key[:,gt_class]=1.0

    # calculate binary cross entropy loss (BCE) and sum up the error for each box (so we get 25 numbers)
    loss = F.binary_cross_entropy(torch.sigmoid(class_pred), ans_key, reduction='none')
    cls_loss=loss.sum(dim=1) # Sum across the 80 classes, resulting in shape [25]

    # box loss (GIoU)
    # calculate the iou between predicted box and ground truth box 
    #the overlap between 25 predicted boxes and 1 gt box
    iou=torchvision.ops.box_iou(box_preds,gt_box) #iou_score is a function that calculates the iou between two boxes
    #iou shape becomes [25,1]
    
    # the cost is calculated as 1 - overlap
    reg_cost=1.0-iou

    # According to YOLOX: Total Cost is the Classification Cost plus 3x the Regression Cost.
    total_cost=cls_loss+(3.0*reg_cost)    # 3.0 multiplier emphasizes the regression accuracy
    
    return total_cost,iou


def calculate_dynamic_k(ious,num_candidates=10):
    # iou:[25] -> the ious from the calculate_loss_matrix
    total_available_cells=ious.shape[0]

    # grab the top 10 highest iou values
    #torch.topk returns the values and the indices we only need the vales[0]
    top_k_values=torch.topk(ious,k=num_candidates)[0]

    # the dynamic k is the sum of the top 10 values
    dynamic_k=top_k_values.sum()

    #we will be rounddind this to nearrest whole number and convert it into int to be used as our dynamic k
    dynamic_k=torch.round(dynamic_k).int()

    #we must get or take  at least 1 we cannot take or hire more than 25 (the number of candidates)
    dynamic_k=torch.clamp(dynamic_k,min=1,max=25)

    return dynamic_k



