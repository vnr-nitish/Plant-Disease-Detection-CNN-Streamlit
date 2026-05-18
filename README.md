# Plant Disease Detection Using Deep Learning

## Project Overview

This project develops a deep learning-based Plant Disease Detection System capable of identifying plant diseases from leaf images using Convolutional Neural Networks (CNN). The model classifies images into 38 plant disease categories and is integrated with a Streamlit web application for real-time prediction.

The system provides an automated approach for early disease diagnosis, helping farmers and researchers detect plant health issues efficiently and supporting sustainable agricultural practices.

---

## Problem Statement

Plant diseases significantly impact crop productivity and agricultural output. Traditional disease identification often requires manual inspection and domain expertise, making the process time-consuming and less scalable.

The objective of this project is to build and deploy an image classification system capable of accurately detecting plant diseases using computer vision and deep learning techniques.

---

## Dataset Information

**Dataset Used:** Plant Village Dataset

Dataset characteristics:

- Multi-class image dataset
- 38 disease categories
- Healthy and diseased plant classes
- Crop-specific disease labels
- Images resized to **224 × 224**

Dataset Source:

:contentReference[oaicite:0]{index=0}

Sample disease classes:

- Apple Scab
- Tomato Leaf Mold
- Corn Rust
- Tomato Early Blight
- Healthy Plant Leaves

---

## Model Architecture

### Deep Learning Framework

- TensorFlow
- Keras

### CNN Architecture

Implemented layers:

- Conv2D
- MaxPooling2D
- Dropout
- Flatten
- Dense

Model configuration:

| Parameter | Value |
|-----------|--------|
| Input Shape | (224,224,3) |
| Activation Function | ReLU, Softmax |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Classification Type | Multi-class Classification |
| Total Classes | 38 |

Trained model:

:contentReference[oaicite:1]{index=1}

---

## System Workflow

### Step 1: Data Preparation

Performed:

- Image preprocessing
- Image resizing to 224 × 224
- Dataset organization
- Label preparation

### Step 2: Model Development

Implemented:

- CNN architecture design
- Model compilation
- Training process
- Hyperparameter setup

### Step 3: Model Evaluation

Evaluated using:

- Training accuracy
- Validation accuracy
- Loss curves
- Test image predictions

### Step 4: Application Deployment

Integrated the trained model into a Streamlit application.

Implemented:

- Image upload functionality
- Image preview
- Real-time prediction generation
- Disease classification output

---

## Application Features

### Prediction Capabilities

- Upload plant leaf images
- Real-time disease prediction
- Classification across 38 categories
- Instant prediction output

### User Interface Features

- Interactive Streamlit application
- Image preview support
- User-friendly interface
- Fast prediction response

---

## Technologies Used

| Category | Tools |
|-----------|-------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, PIL |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Deployment | Streamlit |

---

## Model Evaluation

Performance analysis includes:

- Training accuracy
- Validation accuracy
- Loss visualization
- Prediction performance on unseen images

---

## Deployment

Live Application:

:contentReference[oaicite:2]{index=2}

---

## Future Improvements

- Add prediction confidence scores
- Integrate Grad-CAM visualization
- Convert model to TensorFlow Lite (TFLite)
- Add multilingual support
- Mobile application deployment
- IoT integration for field-level monitoring

---

## Author

**Vinnakota Nitish Raj**

LinkedIn: :contentReference[oaicite:3]{index=3}
