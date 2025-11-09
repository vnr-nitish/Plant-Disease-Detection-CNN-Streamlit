import os
import io
import glob
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import gdown

# ================== CONFIG ==================
MODEL_FILENAME = "plant_disease_cnn_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ================== SMALL UTILITIES ==================
def _fallback_st_image(img: Image.Image, caption: str = ""):
    """Use new arg first; fall back for older Streamlit builds if needed."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

# ================== BANNER HANDLING ==================
def show_banner():
    """Show banner from local repo (prefer exact Disease.png), else from BANNER_URL secret."""
    # 1) Try exact common names first
    candidates = [
        "Disease.png", "disease.png",
        "Disease.jpg", "disease.jpg",
        "Disease.jpeg", "disease.jpeg",
        "Disease.webp", "disease.webp",
    ]
    for name in candidates:
        if os.path.exists(name):
            try:
                img = _open_rgb(name)
                _fallback_st_image(img)
                return
            except Exception:
                pass

    # 2) Glob fallback (accepts any Disease.* / banner.*)
    for pat in ("Disease.*", "disease.*", "banner.*", "Banner.*"):
        for p in glob.glob(pat):
            if os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                try:
                    img = _open_rgb(p)
                    _fallback_st_image(img)
                    return
                except Exception:
                    pass

    # 3) Optional: remote banner via secrets
    url = st.secrets.get("BANNER_URL", "").strip()
    if url:
        try:
            import requests
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            _fallback_st_image(img)
            return
        except Exception:
            st.info("Banner URL provided but could not be loaded.")

# ================== MODEL LOADING ==================
def ensure_model_local() -> str:
    """
    Downloads the Keras model from Google Drive once (via Secrets -> MODEL_DRIVE_ID).
    """
    file_id = st.secrets.get("MODEL_DRIVE_ID", "").strip()
    if not file_id:
        raise RuntimeError(
            "MODEL_DRIVE_ID not set. In Streamlit Cloud → Settings → Secrets, add:\n"
            'MODEL_DRIVE_ID = "1gTUllvoZMRP4HBXMK7TBv2YhqZHXVjyl"'
        )
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading model from Google Drive… (first run only)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_FILENAME, quiet=False)
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError("Model download failed. Check Drive permissions and file id.")
    return MODEL_FILENAME

@st.cache_resource(show_spinner=True)
def load_model_cached():
    path = ensure_model_local()
    # Some older-saved models may need compile=False (Keras format differences)
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return tf.keras.models.load_model(path, compile=False)

# ================== INFERENCE ==================
def preprocess(pil_img: Image.Image) -> np.ndarray:
    # Ensure RGB, resize, normalize
    arr = np.asarray(pil_img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)

def predict(pil_img: Image.Image) -> int:
    model = load_model_cached()
    x = preprocess(pil_img)
    y = model.predict(x, verbose=0)
    return int(np.argmax(y, axis=-1)[0])

# ================== UI ==================
st.set_page_config(page_title="Plant Disease Prediction", page_icon="🌿", layout="centered")

st.sidebar.title("Plant Disease Prediction")
page = st.sidebar.selectbox("Select page", ["Home", "Disease Recognition"])

# Banner (local file or secrets URL)
show_banner()

st.markdown(
    "<h1 style='text-align:center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>",
    unsafe_allow_html=True
)

if page == "Home":
    st.write(
        "- Upload a leaf image on **Disease Recognition** to get a prediction.\n"
        "- The model (~299 MB) is downloaded from Google Drive on first run and cached.\n"
        "- If the banner doesn’t show, ensure `Disease.png` exists at the repo root, "
        "or set a public `BANNER_URL` in Secrets."
    )

else:
    up = st.file_uploader("Choose a leaf image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if up:
        try:
            img_raw = Image.open(up)              # may be CMYK/RGBA/etc.
            img = img_raw.convert("RGB")          # force RGB for consistency
            _fallback_st_image(img, caption=str(getattr(up, "name", "uploaded image")))
            st.caption(f"Mode: {img_raw.mode}, size: {img_raw.size[0]}×{img_raw.size[1]}")

            if st.button("Predict"):
                with st.spinner("Analyzing…"):
                    idx = predict(img)
                    st.success(f"Prediction: {CLASS_NAMES[idx]}")
        except UnidentifiedImageError:
            st.error("Could not read the image. Please upload a JPG/PNG file.")
        except Exception as e:
            st.error(f"Failed to process the image: {type(e).__name__}: {e}")
            st.info("Tip: Try another JPG/PNG. If it was HEIC/CMYK, convert to RGB PNG/JPG and retry.")
    else:
        st.info("Upload an image to begin.")
