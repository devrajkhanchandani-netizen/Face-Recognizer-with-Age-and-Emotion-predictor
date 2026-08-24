# Real-Time Face, Age & Emotion Detector

A simple Python tool that grabs your webcam feed, detects faces in real-time, and predicts both estimated age and dominant emotion on the fly.

## What it uses

* **Face Detection:** OpenCV DNN module using the pre-trained SSD Caffe model.
* **Age Estimation:** Custom regression model built with TensorFlow/Keras (`train.py` + `generalized_age_model.keras`).
* **Emotion Recognition:** `DeepFace` library.

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone [https://github.com/devrajkhanchandani-netizen/Face-Recognizer-with-Age-and-Emotion-predictor.git](https://github.com/devrajkhanchandani-netizen/Face-Recognizer-with-Age-and-Emotion-predictor.git)
cd Face-Recognizer-with-Age-and-Emotion-predictor
```

### 2. Install Requirements
```bash
pip install opencv-python numpy tensorflow deepface
```

## Usage
Run the main script to start the webcam feed:
```bash
python main.py
```

## Project Structure

```text
Face-Recognition-with-Age-and-Emotion-predictor/
├── models/
│   ├── deploy.prototxt                          # OpenCV Caffe face detector config
│   ├── res10_300x300_ssd_iter_140000.caffemodel # OpenCV Caffe face detector weights
│   ├── generalized_age_model.keras              # Age regression model weights
│   └── custom_age_model.keras                   # Alternate/custom age weights
├── generalized_training_colab.ipynb             # Model training notebook
├── train.py                                     # Model architecture definition
├── main.py                                      # Main real-time detection script
├── requirements.txt
└── README.md
```

## License & Credits
* OpenCV for DNN face detection models.
* DeepFace by Serengil for emotion detection backend.
