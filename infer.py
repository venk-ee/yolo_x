"""
YOLO-X Inference Script
Usage: python infer.py <image_path> [--weights path] [--conf 0.25] [--iou 0.5] [--save output.jpg] [--show]
"""

import argparse
import sys
import cv2
import torch
from model import YOLOX
from utils import post_process_nms
from data import letterbox_image, unletterbox_boxes_xyxy

# Pascal VOC class names (20 classes)
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# One color per class for drawing
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-X Inference")
    p.add_argument("image", nargs="?", default="assects/image_for_infer_detected.jpg", help="Path to input image")
    p.add_argument("--weights", default="assects/best_model.pth", help="Path to model weights")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--save", default=None, help="Output path (default: <name>_detected.jpg)")
    p.add_argument("--show", action="store_true", help="Display result in a window")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    num_classes = len(CLASS_NAMES)
    model = YOLOX(num_classes=num_classes)

    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    # Handle both raw state_dict and checkpoint dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # --- Read image ---
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image '{args.image}'")
        sys.exit(1)

    orig_h, orig_w = img.shape[:2]

    # --- Preprocess (letterbox to 640x640) ---
    resized, scale, pad_left, pad_top, _, _ = letterbox_image(img)
    tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).float()
    tensor = tensor.permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        outputs = model(tensor)

    results = post_process_nms(outputs, score_thresh=args.conf, iou_thresh=args.iou)
    boxes, scores, labels = results[0]

    # --- Map boxes back to original image coords ---
    boxes = unletterbox_boxes_xyxy(boxes, scale, pad_left, pad_top, orig_w, orig_h)

    # --- Draw detections ---
    for box, score, label_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        if x2 <= x1 or y2 <= y1:
            continue

        cls_idx = int(label_id.item())
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
        color = COLORS[cls_idx % len(COLORS)]
        text = f"{cls_name} {score:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Save / Show ---
    if args.save:
        save_path = args.save
    else:
        import os
        name = os.path.splitext(os.path.basename(args.image))[0]
        save_path = f"{name}_detected.jpg"

    cv2.imwrite(save_path, img)
    print(f"Found {len(boxes)} detections. Saved to: {save_path}")

    if args.show:
        cv2.imshow("YOLO-X Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
