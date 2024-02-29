import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from model import infer

# Load the pre-trained model
model = tf.keras.models.load_model('LLIE.h5', compile=False)

# Function to enhance image
def enhance_image(image):
    enhanced_image = infer(image)
    return enhanced_image

# Streamlit app
def main():
    st.title("Image Enhancement")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)
        st.markdown("---")

        if st.button("Enhance"):
            enhanced_image = enhance_image(original_image)
            st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

if __name__ == "__main__":
    main()

