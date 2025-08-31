import os
import streamlit as st
from PIL import Image
import numpy as np

# Utility: Model summary string
def get_model_info(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(list(model.modules()))
    return f"Model Summary: {n_layers} layers, {n_params:,} parameters"

# Suppress torch.meshgrid deprecation warning
import warnings
warnings.filterwarnings('ignore', message='.*torch.meshgrid: in an upcoming release.*')

# YOLOv7 inference wrapper
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov7"))  # robust path for yolov7 modules

import torch
# Clear any cached modules for a fresh import
import importlib
for mod in ["models.experimental", "utils.general", "utils.datasets"]:
    if mod in sys.modules:
        del sys.modules[mod]
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

st.set_page_config(
    page_title="Fish Detector — YOLOv7",
    page_icon="🐟",
    layout="wide"
)
st.title("🐟 Fish Detector — YOLOv7")

# Sidebar sliders for confidence and IoU
with st.sidebar:
    st.header("Settings")
    st.warning("Due to limited training resources, model's best accuracy is around 0.50 confidence. If fish are not being detected, try lowering the confidence threshold to around 0.50.")
    conf = st.slider("Confidence", 0.05, 0.95, 0.50, 0.01)
    iou = st.slider("IoU (Intersection over Union)", 0.1, 0.9, 0.45, 0.01,
        help="IoU measures bounding box overlap. Higher value (e.g. 0.9) = stricter matching (fewer but more precise detections). Lower value (e.g. 0.1) = more lenient (more detections but might include some overlapping boxes).")
    imgsz = 640

@st.cache_resource
def load_model(weights_path):
    model = attempt_load(weights_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def infer_image(model, img, conf_thres=0.25, iou_thres=0.45, img_size=640):
    import cv2
    img0 = img.copy()
    img = letterbox(img, img_size, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
    labels = []
    # Try to get class names from model
    class_names = getattr(model, 'names', None)
    if class_names is None and hasattr(model, 'module') and hasattr(model.module, 'names'):
        class_names = model.module.names
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf_score, cls in pred:
            xyxy = [int(x.item()) for x in xyxy]
            # Map class index to name if available
            class_idx = int(cls.item())
            label = class_names[class_idx] if class_names and class_idx < len(class_names) else str(class_idx)
            labels.append(label)
            # Draw bounding box
            img0 = cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 96, 100), 2)
            # Draw filled label box
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top_left = (xyxy[0], max(xyxy[1] - th - baseline, 0))
            bottom_right = (xyxy[0] + tw + 6, xyxy[1])
            cv2.rectangle(img0, top_left, bottom_right, (224, 247, 250), -1)  # filled box (BGR)
            cv2.rectangle(img0, top_left, bottom_right, (178, 235, 242), 1)   # border
            # Draw label text
            cv2.putText(img0, label, (xyxy[0] + 3, xyxy[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 96, 100), 2)
    return img0, labels


# Use absolute path for weights to avoid YOLOv7 download bug
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "brackish.pt")
if not os.path.exists(weights_path):
    st.error(f"Weights not found at {weights_path}. Please ensure the model weights are available at this location.")
    st.stop()

model = load_model(weights_path)

# Show model summary in the UI
st.info(get_model_info(model))


import random
import glob

def get_random_test_image():
    # Use only the /test/urpc directory in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, "test", "urpc")
    images = glob.glob(os.path.join(test_dir, "*.jpg"))
    if not images:
        st.error(f"No test images found in {test_dir} directory.")
        return None, None
    img_path = random.choice(images)
    img = Image.open(img_path).convert("RGB")
    return img, os.path.basename(img_path)

option = st.radio("Select input method", ("Random Test Image", "Upload Image"), index=0)

if option == "Random Test Image":
    if st.button("Test Random Image", help="Click to test a random image from /test"):
        img, fname = get_random_test_image()
        if img is not None:
            img_np = np.array(img)
            result, labels = infer_image(model, img_np, conf_thres=conf, iou_thres=iou, img_size=imgsz)
            cols = st.columns(2)
            with cols[0]:
                st.image(img, caption="Original", use_column_width=True)
            with cols[1]:
                st.image(result, caption="Detection Result", use_column_width=True)
            if labels:
                from collections import Counter
                st.markdown("**Detected Items:**")
                label_counts = Counter(labels)
                summary = [f"{count} x {label}" for label, count in label_counts.items()]
                st.markdown(
                    "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>" +
                    "".join([
                        f"<span style='background:#fff; color:#222; border-radius:6px; padding:4px 10px; font-size:1em; border:1px solid #888;'>{s}</span>"
                        for s in summary
                    ]) +
                    "</div>", unsafe_allow_html=True)

else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        result, labels = infer_image(model, img_np, conf_thres=conf, iou_thres=iou, img_size=imgsz)
        cols = st.columns(2)
        with cols[0]:
            st.image(img, caption="Original", use_column_width=True)
        with cols[1]:
            st.image(result, caption="Detection Result", use_column_width=True)
        if labels:
            from collections import Counter
            st.markdown("**Detected Items:**")
            label_counts = Counter(labels)
            summary = [f"{count} x {label}" for label, count in label_counts.items()]
            st.markdown(
                "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>" +
                "".join([
                    f"<span style='background:#fff; color:#222; border-radius:6px; padding:4px 10px; font-size:1em; border:1px solid #888;'>{s}</span>"
                    for s in summary
                ]) +
                "</div>", unsafe_allow_html=True)

# Always show 'What can I detect?' expander at the very end of the app
with st.expander("🎣 What can I detect?", expanded=True):
    st.markdown("""
    ### 🔍 This AI can spot these underwater classes:

    | Class Index | Name         |
    |-------------|--------------|
    | 0           | echinus      |
    | 1           | holothurian  |
    | 2           | scallop      |
    | 3           | starfish     |

    > 💡 **Tip**: For best results, use clear underwater images with good lighting!
    """)
