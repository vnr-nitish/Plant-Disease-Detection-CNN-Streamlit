# 🌿 Plant Disease Classification Using CNN + Streamlit  
**Author:** Nitish Raj Vinnakota | [LinkedIn](https://linkedin.com/in/vnr-nitish)

---

## 🔍 Project Overview

This project builds a **Convolutional Neural Network (CNN)** that accurately classifies plant diseases based on leaf images. It further integrates the model into a **Streamlit web application** that allows users to upload an image and get instant predictions.

With over 38 plant disease categories, this system provides a practical solution to aid farmers and researchers in diagnosing plant diseases early, promoting **sustainable agriculture**.

---

## 🎯 Problem Statement

> Design and deploy an image classification model that can detect a variety of plant diseases across multiple crops with high accuracy using computer vision.

---

## 🌱 Dataset

- **Type:** Plant Village Dataset (multi-class image dataset)  
- **Classes:** 38 different categories (e.g. Tomato Leaf Mold, Apple Scab, Corn Rust, etc.)  
- **Labels:** Healthy vs diseased (crop-specific)  
- **Images:** Preprocessed to `224x224` resolution

---

## 🧠 Model Architecture

- **Framework:** TensorFlow / Keras  
- **Model:** Custom CNN  
  - Conv2D → MaxPooling → Dropout → Flatten → Dense  
- **Activation:** ReLU, Softmax  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Input Shape:** `(224, 224, 3)`

---

## 🧪 Evaluation Metrics

- Training Accuracy  
- Validation Accuracy  
- Loss Curves  
- Test accuracy over unseen images

---

## 🧰 Tech Stack

- `Python`  
- `TensorFlow / Keras`  
- `NumPy`, `OpenCV`, `PIL`  
- `Streamlit` for deployment  
- `Matplotlib` for visualization  

---

## 🚀 Features of Streamlit Web App

- 📤 Upload an image of a diseased leaf  
- ⚙️ CNN model classifies it into one of 38 categories  
- ✅ Prediction displayed with disease name  
- 🖼️ Optional image preview

### ✅ How to Run the App:

```bash
# Install required libraries
pip install -r requirements.txt

# Run the app
streamlit run app.py
