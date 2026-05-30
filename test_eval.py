import torch
from utils import get_data_loader
from model import YOLOX
from utils import post_process_nms, format_for_coco


val_data_loader = get_data_loader(train=False, val=True, test=False, BATCH_SIZE=1)[1]
device = torch.device("cuda")
model = YOLOX(num_classes=80).to(device)
model.eval()


for images, targets in val_data_loader:
    images = images.to(device)
    with torch.no_grad():
        out = model(images)

    results = post_process_nms(out)
    preds = format_for_coco(results, targets)

    print("GT BOXES (orig_size):")
    print(targets[0]["orig_size"])
    print(
        targets[0]["boxes"]
    )  # Wait, targets[0]['boxes'] is resized boxes in cxcywh format

    print("PREDS for this image:")
    for p in preds[:5]:
        print(p)
    break
