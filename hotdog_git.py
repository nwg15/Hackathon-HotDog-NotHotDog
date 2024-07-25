

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image



#base_dir =  'c:/Users/nkech/Documents/GeneralAssembly/Projects/project-4-HotdogNotHotDog/Hackathon-HotDog-NotHotDog/Model/'
#model_path = base_dir + "modelnew.h5"

# Function to load and preprocess the image
def load_and_prep_image(file, img_shape=299):
    """
    Reads an image from file, turns it into a tensor and reshapes it
    to (img_shape, img_shape, color_channels).
    """
    img = load_img(file, target_size=(img_shape, img_shape))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Ensure the model path is correct and accessible
#model_path = r"C:\Users\nkech\Documents\GeneralAssembly\Projects\project-4-HotdogNotHotDog\Hackathon-HotDog-NotHotDog\Model\modelnew.h5"
model = load_model('model.h5')

# Setting up the Streamlit UI
st.title("Hotdog or Not-Hotdog Classifier")
st.write(
    "This tool uses a neural network to classify images as 'hotdog' or 'not-hotdog'. Upload an image to get started."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = load_and_prep_image(uploaded_file)

    # Displaying the image using PIL
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        prediction = model.predict(image)
        if prediction < 0.5:
            st.write(f"Prediction: Not-Hotdog ({100 * (1 - prediction[0][0]):.2f}%)")
        else:
            st.write(f"Prediction: Hotdog ({100 * prediction[0][0]:.2f}%)")