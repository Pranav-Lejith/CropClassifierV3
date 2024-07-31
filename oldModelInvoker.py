import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

model = tf.keras.models.load_model('crop_classifier_model.h5')

class_labels = {
    0: 'Maize',
    1: 'Wheat',
}

def prepare_image(image):
    image = image.resize((150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

st.title("Crop Classification")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    prepared_image = prepare_image(image)
    
    prediction = model.predict(prepared_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels.get(predicted_class_index, "Unknown")
    
    st.write(f"The predicted class is: {predicted_class}")
