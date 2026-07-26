# Bangladeshi Note and Coin YOLO Detection

A computer vision project for detecting **Bangladeshi Taka banknotes and coins** using the **YOLO object detection model**. The project fine-tunes a pretrained YOLO model on a custom Bangladeshi currency dataset to recognize different denominations from images.

## 📌 Project Overview

This project uses **YOLO26n** with transfer learning to detect Bangladeshi currency notes. The model is first fine-tuned on a banknote dataset and later extended with a coin dataset to create a combined currency detection model.

The main objectives are:

* Detect Bangladeshi Taka notes from images
* Identify different denominations
* Extend detection capability to coins
* Evaluate YOLO performance using standard object detection metrics

---

## 🚀 Technologies Used

* Python
* YOLO (Ultralytics)
* PyTorch
* OpenCV
* Google Colab
* Matplotlib

---

## 📂 Dataset

The dataset contains annotated images of Bangladeshi currency.

### Dataset Structure

```
bankNote_dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

Dataset split:

* Training: 70%
* Validation: 20%
* Testing: 10%

Annotations follow the YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

# 🏦 Banknote Detection

## Model

A pretrained YOLO26n model was used as the base model.

```python
YOLO("yolo26n.pt")
```

Before fine-tuning, the pretrained model was tested on Bangladeshi currency images. Since the model was not trained on Taka currency, it was unable to detect the notes correctly.

---

## Training

The model was fine-tuned using the Bangladeshi Taka note dataset.

Training configuration:

| Parameter  | Value   |
| ---------- | ------- |
| Model      | YOLO26n |
| Epochs     | 50      |
| Image Size | 640×640 |
| Batch Size | 16      |
| Device     | GPU     |

Training command:

```python
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

---

# 📊 Evaluation Results (Banknotes)

The trained model achieved strong detection performance:

| Metric    |  Score |
| --------- | -----: |
| mAP@50    | 0.9456 |
| mAP@50-95 | 0.8265 |
| Precision | 0.9275 |
| Recall    | 0.9365 |

### Performance Analysis

* The model achieved high accuracy in detecting Bangladeshi notes.
* Most denominations showed excellent detection performance.
* Bounding box localization was also accurate, shown by the high mAP@50-95 score.

---

# 🪙 Coin Detection Extension

The trained banknote detector was further fine-tuned using a coin dataset.

Training setup:

| Parameter  | Value               |
| ---------- | ------------------- |
| Base Model | Banknote YOLO Model |
| Epochs     | 25                  |
| Image Size | 640×640             |
| Batch Size | 16                  |

Training:

```python
bonus_model.train(
    data="coin_dataset/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16
)
```

---

## Coin Dataset Issue

During coin model training, some images and labels in the coin dataset were corrupted.

Because of these data quality issues:

* Training performance decreased
* Evaluation metrics were lower compared to the banknote-only model
* The combined note + coin detector achieved reduced accuracy

Improving the dataset quality and removing corrupted annotations would likely improve performance.

---

# 🔍 Inference Example

The trained model can detect currency from new images:

```python
results = trained_model.predict(
    source="test_image.jpg",
    conf=0.25
)
```

The output includes:

* Detected currency class
* Bounding box coordinates
* Confidence score

---

# 📁 Project Structure

```
Bangladeshi-Currency-YOLO/
│
├── notebook.ipynb
├── README.md
│
├── dataset/
│   ├── bankNote_dataset/
│   └── coin_dataset/
│
└── runs/
    └── detect/
        ├── bangladeshi_taka_detector/
        └── bangladeshi_taka_coins_detector/
```

---

# 🎯 Future Improvements

* Collect a larger and more diverse coin dataset
* Remove corrupted images and annotations
* Apply data augmentation
* Train larger YOLO models (YOLO26s/m/l)
* Deploy as a mobile currency recognition application
* Add real-time camera detection

---

# 👨‍💻 Author

**Musfiq Hossain**

Computer Science & Engineering Student
Machine Learning & Computer Vision Enthusiast

---

# 📜 License

This project is intended for educational and research purposes.
