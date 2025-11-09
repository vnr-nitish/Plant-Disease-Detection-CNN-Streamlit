import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

MODEL_FILENAME = "plant_disease_cnn_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']

def ensure_model_local() -> str:
    file_id = st.secrets.get("MODEL_DRIVE_ID", "").strip()
    if not file_id:
        raise RuntimeError(
            "MODEL_DRIVE_ID not set. In Streamlit Cloud → Settings → Secrets, add:\n"
            'MODEL_DRIVE_ID = "1gTUllvoZMRP4HBXMK7TBv2YhqZHXVjyl"'
        )
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading model from Google Drive…")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_FILENAME, quiet=False)
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError("Model download failed. Check Drive permissions and file id.")
    return MODEL_FILENAME

@st.cache_resource(show_spinner=True)
def load_model_cached():
    path = ensure_model_local()
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        # Older-saved models (Keras2 format) sometimes need compile=False
        return tf.keras.models.load_model(path, compile=False)

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(pil_img: Image.Image) -> int:
    model = load_model_cached()
    x = preprocess(pil_img)
    y = model.predict(x, verbose=0)
    return int(np.argmax(y, axis=-1)[0])

st.set_page_config(page_title="Plant Disease Prediction", page_icon="🌿")
st.sidebar.title("Plant Disease Prediction")
page = st.sidebar.selectbox("Select page", ["Home", "Disease Recognition"])

st.markdown("<h1 style='text-align:center;'>Plant Disease Prediction System</h1>", unsafe_allow_html=True)

if page == "Home":
    st.write("Go to **Disease Recognition** and upload a leaf image.")
else:
    up = st.file_uploader("Choose a leaf image (JPG/PNG):", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up)
        st.image(img, caption=up.name, use_container_width=True)
        if st.button("Predict"):
            with st.spinner("Analyzing…"):
                try:
                    idx = predict(img)
                    st.success(f"Prediction: {CLASS_NAMES[idx]}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload an image to begin.")
