
# Fish Detector (YOLOv7)

A production-ready, modular Streamlit app for underwater object detection using YOLOv7, with CLI, Docker, CI/CD, and config support.

---

## Project Structure

```plaintext
├── src/    
│   └── app.py            # Streamlit web app for detection    
├── cli.py                # Command-line interface for batch inference
├── config.yaml           # Config file for model/app settings
├── Dockerfile            # Docker container support
├── .github/workflows/    # GitHub Actions CI/CD
│   └── python-app.yml
├── docs/                 # Documentation starter
│   └── README.md
├── weights/
│   ├── urpc.pt           # URPC model weights
│   └── brackish.pt       # Brackish model weights
├── notebooks/
│   └── train_yolov7_urpc2019.ipynb  # Training notebook
├── requirements.txt      # Python dependencies
├── tests/                # Unit tests
│   └── test_app.py
├── test/
│   ├── urpc/             # URPC test images (test-urpc-#.jpg)
│   └── brackish/         # Brackish test images (test-brackish-#.jpg)
├── yolov7/               # YOLOv7 code (as a dependency, do not edit)
└── README.md             # This file
```

---

## Setup

1. **Python Version**
    - This project requires **Python 3.12**. (See `.streamlit/runtime.txt`)

2. **Create a virtual environment**
        ```bash
        python3.12 -m venv venv
        source venv/bin/activate
        ```

3. **Install dependencies**
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

4. **Add your trained weights**
    - Place your URPC model weights as `urpc.pt` and Brackish model weights as `brackish.pt` in the `weights/` directory.

5. **Run the Streamlit app**
        ```bash
        streamlit run src/app.py
        ```

6. **Test images**
    - Place `.jpg` images in `test/urpc/` (named `test-urpc-#.jpg`) and `test/brackish/` (named `test-brackish-#.jpg`) to use the "Random Test Image" feature for each model.

---

## CLI Usage

Run batch inference from the command line:

```bash
python cli.py --image path/to/image.jpg --weights weights/urpc.pt
python cli.py --image path/to/image.jpg --weights weights/brackish.pt
```

---

## Docker Usage

Build and run the app in a container:

```bash
docker build -t fish-detector-yolov7 .
docker run -p 8501:8501 fish-detector-yolov7

```

---

## CI/CD

Automated tests run on every push via GitHub Actions:

```yaml
name: Python application
on:
    push:
        branches: [ main ]
    pull_request:
        branches: [ main ]
jobs:
    build:
        runs-on: ubuntu-latest
        steps:
        - uses: actions/checkout@v3
        - name: Set up Python 3.12
            uses: actions/setup-python@v4
            with:
                python-version: '3.12'
        - name: Install dependencies
            run: |
                python -m pip install --upgrade pip
                pip install -r requirements.txt
        - name: Run tests
            run: |
                python -m unittest discover tests
```

---

## Config File

Edit `config.yaml` to change model/app settings:

```yaml
model:
    weights: model/best.pt
    conf_threshold: 0.25
    iou_threshold: 0.45
app:
    port: 8501
    theme: light
```

---

## Testing

Run unit tests:

```bash
python -m unittest discover tests
```

---

## Class Labels


### URPC Model Classes
| Class Index | Name         |
|-------------|--------------|
| 0           | echinus      |
| 1           | holothurian  |
| 2           | scallop      |
| 3           | starfish     |

### Brackish Model Classes
| Class Index | Name         |
|-------------|--------------|
| 0           | crab         |
| 1           | fish         |
| 2           | jellyfish    |
| 3           | shrimp       |
| 4           | small_fish   |
| 5           | starfish     |

---

## Features
- Upload or test on random underwater images
- Adjustable confidence and IoU thresholds
- Detected objects are shown with bounding boxes and readable labels
- Summary of detected items (e.g., `5 x Echinus`)
- Batch CLI inference
- Docker container support
- CI/CD with GitHub Actions
- Configurable via YAML
- Unit tests included

---

## Training
- See `notebooks/train_yolov7_urpc2019.ipynb` for a Colab-ready training workflow.
- Training uses YOLOv7 and the URPC2019 dataset.

---

## Notes
- The `yolov7/` folder is a dependency and should not be modified.
- For custom training, follow the notebook and place your weights in `model/best.pt`.

---

## License
- YOLOv7 code: [YOLOv7 License](yolov7/LICENSE.md)
- This app: MIT License