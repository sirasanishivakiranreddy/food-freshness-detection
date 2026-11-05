"""Streamlit interface for fresh/rotten fruit predictions."""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model" / "food_freshness.keras"
TRAINING_ACCURACY_PATH = ROOT_DIR / "model" / "training_accuracy.png"
CONFUSION_MATRIX_PATH = ROOT_DIR / "model" / "confusion_matrix.png"
IMAGE_SIZE = (128, 128)


@st.cache_resource
def load_model():
	return tf.keras.models.load_model(MODEL_PATH)


st.set_page_config(page_title="Food Freshness Detection", page_icon="🍎")
st.title("Food Freshness Detection")
st.write("Upload an apple, banana, or orange image to check its freshness.")

uploaded_file = st.file_uploader(
	"Choose a fruit image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
	try:
		image = Image.open(uploaded_file).convert("RGB")
	except UnidentifiedImageError:
		st.error("This file could not be read as an image. Try another image.")
	else:
		st.image(image, caption="Uploaded image", width="stretch")
		if not MODEL_PATH.is_file():
			st.info("Train the CNN first by running `python train.py`.")
		else:
			model = load_model()
			resized_image = image.resize(IMAGE_SIZE)
			image_array = np.asarray(resized_image, dtype=np.float32)[np.newaxis, ...]
			rotten_probability = float(model.predict(image_array, verbose=0)[0][0])
			fresh_probability = 1.0 - rotten_probability
			prediction = "ROTTEN" if rotten_probability >= 0.5 else "FRESH"

			st.subheader("FOOD FRESHNESS RESULT")
			fresh_column, rotten_column = st.columns(2)
			fresh_column.metric("Fresh", f"{fresh_probability:.1%}")
			rotten_column.metric("Rotten", f"{rotten_probability:.1%}")
			if prediction == "FRESH":
				st.success("FINAL RESULT: FRESH")
			else:
				st.error("FINAL RESULT: ROTTEN")

st.divider()
st.subheader("Model evaluation")
if TRAINING_ACCURACY_PATH.is_file():
	st.image(
		str(TRAINING_ACCURACY_PATH),
		caption="Training and validation accuracy over epochs",
		width="stretch",
	)
else:
	st.info("The training accuracy graph will appear here after running `python train.py`.")

if CONFUSION_MATRIX_PATH.is_file():
	st.image(
		str(CONFUSION_MATRIX_PATH),
		caption="Confusion matrix on the held-out test set",
		width="stretch",
	)
else:
	st.info("The confusion matrix will appear here after running `python train.py`.")
