import torch

from utils import post_process_nms, format_for_coco, eval_metrics_coco_bbox


def train_one_epoch(model, dataloader, criterion, device, optimizer, epoch=0, total_epochs=0):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    print(f"\n{'='*60}")
    print(f"  EPOCH {epoch+1}/{total_epochs} — Training")
    print(f"{'='*60}")

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        gt_boxes_list = [t["boxes"].to(device) for t in targets]
        gt_cls_list = [t["labels"].to(device) for t in targets]

        out = model(images)

        loss = criterion(out, gt_boxes_list, gt_cls_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            print(f"  [Batch {batch_idx+1}/{num_batches}]  loss: {loss.item():.4f}  avg_loss: {avg_loss:.4f}")

    print(f"  Train Loss (avg): {avg_loss:.4f}")
    return avg_loss


def val_one_epoch(model, val_loader, criterion, device, epoch=0, total_epochs=0):
    all_preds = []
    running_loss = 0.0
    num_batches = len(val_loader)

    model.eval()

    print(f"\n{'-'*60}")
    print(f"  EPOCH {epoch+1}/{total_epochs} — Validation")
    print(f"{'-'*60}")

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            gt_boxes_list = [t["boxes"].to(device) for t in targets]
            gt_cls_list = [t["labels"].to(device) for t in targets]

            out = model(images)
            loss = criterion(out, gt_boxes_list, gt_cls_list)

            running_loss += loss.item()

            results = post_process_nms(out)

            batch_preds = format_for_coco(results, targets)

            all_preds.extend(batch_preds)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"  [Batch {batch_idx+1}/{num_batches}]  val_loss: {avg_loss:.4f}")

        print(f"\n  COCO Evaluation:")
        mAP = eval_metrics_coco_bbox(val_loader.dataset.coco, all_preds)
    return mAP

