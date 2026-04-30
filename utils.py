import torch
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
