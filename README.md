# Bangladeshi Note and Coin YOLO Detection

A computer vision project for detecting Bangladeshi currency notes and coins using the YOLO object detection framework. The repository is organized around notebook-driven experimentation and training artifacts.

## Core Concepts

- **Object Detection with YOLO**
  - Uses YOLO-based models to localize and classify currency objects (notes and coins) in images.
  - Detection outputs typically include bounding boxes, class labels, and confidence scores.

- **Domain-Specific Dataset Design**
  - Separate datasets for notes and coins suggest a task decomposition strategy:
    - `bankNote_dataset/` for note images/labels
    - `coin_dataset/` for coin images/labels
  - This can help improve labeling consistency and allow targeted experiments.

- **Notebook-Centric Workflow**
  - Development appears to be managed primarily through `Assignment16.ipynb`.
  - Typical workflow in the notebook likely includes:
    1. Environment and dependency setup
    2. Data preparation / configuration
    3. Model training
    4. Validation and inference
    5. Result visualization

- **Experiment Tracking via YOLO Runs**
  - `runs/` contains generated training/inference artifacts (e.g., metrics plots, checkpoints, predictions).
  - Useful for comparing model versions and debugging performance.

- **Transfer Learning / Pretrained Weights**
  - Presence of `yolo26n.pt` indicates use of pretrained weights as a starting point for fine-tuning.

## Repository Structure

```text
.
├── Assignment16.ipynb        # Main notebook for experimentation, training, and evaluation
├── bankNote_dataset/         # Dataset related to banknote detection
├── coin_dataset/             # Dataset related to coin detection
├── runs/                     # YOLO training/inference outputs and experiment artifacts
└── yolo26n.pt                # YOLO pretrained/model weight file
```

## Typical Pipeline

1. **Prepare Data**
   - Organize images and annotations in YOLO-compatible format.
   - Define dataset configuration (class names and paths).

2. **Train / Fine-Tune Model**
   - Start from pretrained weights (`yolo26n.pt`) when appropriate.
   - Adjust training hyperparameters (epochs, image size, batch size).

3. **Evaluate Performance**
   - Review precision/recall, mAP, and qualitative predictions from `runs/`.

4. **Run Inference**
   - Detect notes/coins on new images and inspect confidence and bounding boxes.

## Use Cases

- Currency denomination assistance
- Educational demos for object detection
- Foundation for retail/payment automation prototypes

## Requirements (Suggested)

Because this project is notebook-driven, dependencies are likely managed inside the notebook. A typical setup for YOLO workflows includes:

- Python 3.9+
- Jupyter Notebook / JupyterLab
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib

Example install:

```bash
pip install ultralytics opencv-python numpy matplotlib jupyter
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Md-Musfiq-Hossain/Bangladeshi-Note-and-Coin-YOLO-detection.git
cd Bangladeshi-Note-and-Coin-YOLO-detection
```

2. Launch Jupyter:

```bash
jupyter notebook
```

3. Open and run:
   - `Assignment16.ipynb`

## Notes

- Keep large model files and generated run artifacts under control for repository size management.
- For reproducibility, consider adding:
  - `requirements.txt`
  - explicit dataset YAML/config files
  - model/version notes for best checkpoints

## Future Improvements

- Add a clear class list for note and coin denominations.
- Add reproducible training commands and hyperparameters.
- Add a lightweight inference script (CLI) outside notebook.
- Add sample predictions in `README` for quick project preview.

---

If you want, I can next create:
- a `requirements.txt`,
- a `.gitignore` tuned for YOLO/Jupyter,
- and a small `inference.py` script to run predictions from terminal.
