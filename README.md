# 🌿 Plant Disease Detection Using CNN + Streamlit  
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
- **Dataset:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

---

## 🧠 Model Architecture

- **Framework:** TensorFlow / Keras  
- **Model:** Custom CNN  
  - Conv2D → MaxPooling → Dropout → Flatten → Dense  
- **Activation:** ReLU, Softmax  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Input Shape:** `(224, 224, 3)`
- **Model:** https://drive.google.com/file/d/1gTUllvoZMRP4HBXMK7TBv2YhqZHXVjyl/view?usp=share_link

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

---

### ✅ How to Run the App:

 Install required libraries
pip install -r requirements.txt

 Run the app
streamlit run app.py

---

## 🚀 Future Improvements

- 🌐 Deploy the app using **Streamlit Cloud**, **Render**, or **Hugging Face Spaces**
- 📊 Add **confidence scores** and visual explanations using Grad-CAM
- 📱 Convert the model to **TensorFlow Lite (TFLite)** for mobile deployment
- 🌍 Add multilingual support to increase accessibility for farmers worldwide
- ⚙️ Integrate with APIs or IoT devices for field-level prediction and alerts

---

## 📫 Contact

For collaborations, suggestions, or queries, feel free to reach out:

- 📧 **Email:** nvinnako2@gitam.in  
- 🔗 **LinkedIn:** [linkedin.com/in/vnr-nitish](https://linkedin.com/in/vnr-nitish)

---

> *"Empowering agriculture with AI — one leaf at a time."*
