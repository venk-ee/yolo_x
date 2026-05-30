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


best_map = 0

for epoch in range(EPOCHS):
    train_one_epoch(model, train_data_loader, criterion, device, optimizer)
    mAP = val_one_epoch(model, val_data_loader, criterion, device)
    scheduler.step()

    if mAP > best_map:
        best_map = mAP
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best mAP: {best_map:.4f} — model saved!")

print(f"\nTraining complete. Best mAP: {best_map:.4f}")
