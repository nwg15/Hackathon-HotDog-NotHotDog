import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the model
model = load_model('adopted.keras')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize to [0, 1] range
    return img

# Define the Streamlit app
st.title("Hotdog or Not Hotdog Classifier")
st.write("Upload an image to find out if it's a hotdog or not!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make prediction
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)
    
    # Interpret the result
    if prediction[0] > 0.5:
        st.write("It's a hotdog! 🌭")
    else:
        st.write("It's not a hotdog. ❌")
