
# Logo Detection (YOLOv7)

A production-ready, modular Streamlit app for underwater object detection using YOLOv7, with CLI, Docker, CI/CD, and config support.

---

## Project Structure

```plaintext
logo-detection/
├── .src/    
│   └── app.py            # Streamlit web app for detection    
├── cli.py                # Command-line interface for batch inference
├── config.yaml           # Config file for model/app settings
├── Dockerfile            # Docker container support
├── .github/workflows/    # GitHub Actions CI/CD
│   └── python-app.yml
├── docs/                 # Documentation starter
│   └── README.md
├── model/
│   └── best.pt           # Trained YOLOv7 weights
├── notebooks/
│   └── train_yolov7_urpc2019.ipynb  # Training notebook
├── requirements.txt      # Python dependencies
├── tests/                # Unit tests
│   └── test_app.py
├── test/                 # Test images for demo
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
        - Place your YOLOv7 weights as `model/best.pt`.

5. **Run the Streamlit app**
        ```bash
        streamlit run src/app.py
        ```

6. **Test images**
        - Place `.jpg` images in the `test/` folder to use the "Random Test Image" feature.

---

## CLI Usage

Run batch inference from the command line:

```bash
python cli.py --image path/to/image.jpg --weights model/best.pt
```

---

## Docker Usage

Build and run the app in a container:

```bash
docker build -t logo-detection .
docker run -p 8501:8501 logo-detection
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

| Class Index | Name         |
|-------------|--------------|
| 0           | echinus      |
| 1           | holothurian  |
| 2           | scallop      |
| 3           | starfish     |

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