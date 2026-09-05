# AI-ASL-Translater

This project uses a deep learning model to recognize American Sign Language (ASL) hand signs from webcam images or uploaded images. The model uses transfer learning with MobileNetV2 to classify hand signs and provides the prediction with a confidence score and text-to-speech output. This is a current prototype

## Features

- Detects ASL hand signs using a trained deep learning model.
- Supports two input methods:
  - Webcam
  - Image upload
- Uses MobileNetV2 transfer learning for image classification.
- Displays the predicted sign and confidence score.
- Converts the predicted sign into speech.
- Includes drag-and-drop support for image uploads.
- Uses data augmentation during training.

## Model Training

The model is trained using the ASL Alphabet dataset.

- **Image Size:** 128 × 128
- **Batch Size:** 32
- **Epochs:** 10
- **Base Model:** MobileNetV2
- **Optimizer:** Adam
- **Learning Rate:** 0.0005
- **Loss Function:** Categorical Crossentropy

The training process uses rotation, zoom, shifting, shearing, and horizontal flipping for data augmentation.

## Web Application

The Flask web application allows users to:

- Select webcam or upload mode.
- Capture an ASL sign using a webcam.
- Upload an image of an ASL sign.
- Send the image to the trained model.
- Display the predicted sign and confidence.
- Generate speech for the predicted sign.

## Files

- `asl_client.html` — Web interface for the ASL predictor.
- `app.py` — Flask server and prediction system.
- `train.py` — Trains the MobileNetV2 model.
- `asl_transfer_model.keras` — Trained ASL classification model.
- `classes.npy` — Stores the ASL class labels.

## Dependencies

- `tensorflow`
- `numpy`
- `opencv-python`
- `flask`
- `Pillow`
- `gTTS`
- `matplotlib`
