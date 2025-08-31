import argparse
from src.model import load_model
from src.frame_processing import process_image
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv7 Logo Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="model/best.pt", help="Path to model weights")
    args = parser.parse_args()

    model = load_model(args.weights)
    img = Image.open(args.image)
    results = process_image(model, img)
    print("Detections:", results)
