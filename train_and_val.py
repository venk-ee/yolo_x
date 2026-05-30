import torch

from utils import post_process_nms, format_for_coco, eval_metrics_coco_bbox


def train_one_epoch(model, dataloader, criterion, device, optimizer):
    model.train()

    for images, targets in dataloader:
        images = images.to(device)
        gt_boxes_list = [t["boxes"].to(device) for t in targets]
        gt_cls_list = [t["labels"].to(device) for t in targets]

        out = model(images)

        loss = criterion(out, gt_boxes_list, gt_cls_list)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def val_one_epoch(model, val_loader, criterion, device):
    all_preds = []

    model.eval()

    with torch.inference_mode():
        for images, targets in val_loader:
            images = images.to(device)
            gt_boxes_list = [t["boxes"].to(device) for t in targets]
            gt_cls_list = [t["labels"].to(device) for t in targets]

            out = model(images)
            loss = criterion(out, gt_boxes_list, gt_cls_list)
            print(loss)

            results = post_process_nms(out)

            batch_preds = format_for_coco(results, targets)

            all_preds.extend(batch_preds)

        mAP = eval_metrics_coco_bbox(val_loader.dataset.coco, all_preds)
    return mAP
