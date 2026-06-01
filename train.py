from model import YOLOX
import torch
import argparse
import os
import csv
import time
from loss import YOLOXLoss
from utils import get_data_loader, get_optimizer, get_scheduler, get_devices
from train_and_val import train_one_epoch, val_one_epoch

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

train_data_loader, val_data_loader, test_data_loader = get_data_loader(
    BATCH_SIZE=BATCH_SIZE
)

device = get_devices()
if device == "cuda":
    torch.backends.cudnn.benchmark = True

num_classes = len(train_data_loader.dataset.cat_ids)
model = YOLOX(num_classes=num_classes).to(device)

criterion = YOLOXLoss(num_classes=num_classes)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer, epochs=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


SAVE_DIR = "/home/kenny/pytorch/yolo_x/model"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = f"{SAVE_DIR}/metrics.csv"

best_map = 0

with open(CSV_PATH, "w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "mAP",
            "best_mAP",
            "lr",
            "epoch_time_sec",
        ],
    )
    writer.writeheader()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_data_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            total_epochs=EPOCHS,
        )
        mAP, val_loss = val_one_epoch(
            model, val_data_loader, criterion, device, epoch=epoch, total_epochs=EPOCHS
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(f"  ★ New best mAP: {best_map:.4f} — best model saved!")

        epoch_time = time.time() - epoch_start

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_map": best_map,
                "mAP": mAP,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_sec": epoch_time,
            },
            f"{SAVE_DIR}/last_checkpoint.pth",
        )
        print(f"Checkpoint saved.")

        writer.writerow(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mAP": mAP,
                "best_mAP": best_map,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
            }
        )
        csv_file.flush()

        print(
            f"\n  Epoch {epoch+1}/{EPOCHS} Summary:  mAP={mAP:.4f}  best_mAP={best_map:.4f}  lr={current_lr:.6f}"
        )
        print(f"{'='*60}")

print(f"\nTraining complete. Best mAP: {best_map:.4f}")
