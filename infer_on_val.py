import os, sys, cv2, torch
from model import YOLOX
from utils import post_process_nms
from data import letterbox_image, unletterbox_boxes_xyxy

# 1. Setup path & device
image_path = (
    sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip("'\" ")
)
weights_path = "/home/kenny/pytorch/yolo_x/model/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model
model = YOLOX(num_classes=1)
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict.get("model_state_dict", state_dict))
model.to(device).eval()

# 3. Read and Preprocess
img = cv2.imread(image_path)
if img is None:
    print("Invalid image path")
    sys.exit(1)
h, w = img.shape[:2]
resized, scale, pad_left, pad_top, _, _ = letterbox_image(img)
tensor = (
    torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)
    / 255.0
)
tensor = tensor.unsqueeze(0).to(device)

# 4. Inference & Postprocess
with torch.no_grad():
    outputs = model(tensor)
boxes, scores, _ = post_process_nms(outputs, score_thresh=0.25)[0]
boxes = unletterbox_boxes_xyxy(boxes, scale, pad_left, pad_top, w, h)

# 5. Draw and Save
for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = [val.item() for val in box]
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    if x2 <= x1 or y2 <= y1:
        continue

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 235, 0), 2)
    cv2.putText(
        img,
        f"detected {score:.1%}",
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 235, 0),
        2,
        cv2.LINE_AA,
    )

save_path = f"{os.path.basename(os.path.splitext(image_path)[0])}_detected.jpg"
cv2.imwrite(save_path, img)
print(f"Detections found: {len(boxes)}. Result saved to: {save_path}")
