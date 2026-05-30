from model import YOLOX
import torch
from loss import YOLOXLoss
from utils import get_data_loader, get_optimizer, get_scheduler, get_devices
from train_and_val import train_one_epoch, val_one_epoch

EPOCHS = 150
BATCH_SIZE = 4

train_data_loader, val_data_loader, test_data_loader = get_data_loader(
    BATCH_SIZE=BATCH_SIZE
)

device = get_devices()

model = YOLOX(num_classes=80).to(device)

criterion = YOLOXLoss(num_classes=80)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer, epochs=EPOCHS)


SAVE_DIR = "/kaggle/working"

best_map = 0

for epoch in range(EPOCHS):
    train_one_epoch(model, train_data_loader, criterion, device, optimizer, epoch=epoch, total_epochs=EPOCHS)
    mAP = val_one_epoch(model, val_data_loader, criterion, device, epoch=epoch, total_epochs=EPOCHS)
    scheduler.step()

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map": best_map,
        "mAP": mAP,
    }, f"{SAVE_DIR}/last_checkpoint.pth")
    print(f"  Checkpoint saved.")

    # (local only, lost on session end) ---
    # if mAP > best_map:
    #     best_map = mAP
    #     torch.save(model.state_dict(), "best_model.pth")
    #     print(f"New best mAP: {best_map:.4f} — model saved!")

    #  (persists to /kaggle/working/) ---
    if mAP > best_map:
        best_map = mAP
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
        print(f"  ★ New best mAP: {best_map:.4f} — best model saved!")

    print(f"\n  Epoch {epoch+1}/{EPOCHS} Summary:  mAP={mAP:.4f}  best_mAP={best_map:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*60}")

print(f"\nTraining complete. Best mAP: {best_map:.4f}")
