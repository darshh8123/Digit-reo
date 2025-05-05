import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")
st.title("🔢 Handwritten Digit Recognizer")
st.write("Upload an image of a digit (0–9) for prediction using a trained CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing for MNIST model
    image = ImageOps.grayscale(image)      # Convert to grayscale
    image = ImageOps.invert(image)         # Invert to white digit on black background
    image = image.resize((28, 28))         # Resize to 28x28
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Load the model
    model = tf.keras.models.load_model("digit_model.h5")
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)

    st.success(f"🧠 Predicted Digit: **{digit}**")

