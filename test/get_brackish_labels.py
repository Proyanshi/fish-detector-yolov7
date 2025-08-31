import sys
import os
import torch

# Add yolov7 to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov7"))
from models.experimental import attempt_load

# Path to brackish weights
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "brackish.pt")

model = attempt_load(weights_path, map_location=torch.device('cpu'))

# Print class names
if hasattr(model, 'names'):
    class_names = model.names
elif hasattr(model, 'module') and hasattr(model.module, 'names'):
    class_names = model.module.names
else:
    class_names = None

print("Class names in brackish.pt:")
if class_names:
    for idx, name in enumerate(class_names):
        print(f"{idx}: {name}")
else:
    print("No class names found in model.")
