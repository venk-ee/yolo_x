import torch


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
    model.eval()

    with torch.inference_mode():
        for images, targets in val_loader:
            images = images.to(device)
            gt_boxes_list = [t["boxes"].to(device) for t in targets]
            gt_cls_list = [t["labels"].to(device) for t in targets]

            out = model(images)
            loss = criterion(out, gt_boxes_list, gt_cls_list)
            print(loss)
