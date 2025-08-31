import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image

# Add yolov7 to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov7"))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

def infer_image(model, img, conf_thres=0.10, iou_thres=0.10, img_size=640):
    import cv2
    img0 = img.copy()
    img = letterbox(img, img_size, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
    labels = []
    class_names = getattr(model, 'names', None)
    if class_names is None and hasattr(model, 'module') and hasattr(model.module, 'names'):
        class_names = model.module.names
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf_score, cls in pred:
            class_idx = int(cls.item())
            label = class_names[class_idx] if class_names and class_idx < len(class_names) else str(class_idx)
            labels.append(label)
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv7 Fish Detector CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/urpc.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.10, help="IoU threshold")
    args = parser.parse_args()

    model = attempt_load(args.weights, map_location=torch.device('cpu'))
    model.eval()
    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    labels = infer_image(model, img_np, conf_thres=args.conf, iou_thres=args.iou)
    print("Detections:", labels)
