import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="crop_classifier_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Updated class labels to include Cotton and Gram
class_labels = {
    0: 'Wheat',
    1: 'Maize',
    2: 'Cotton',
    3: 'Gram'
}

def prepare_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image.astype(np.float32)

# Theme switcher
theme = st.sidebar.radio("Choose Theme", ("Light", "Dark"))

if theme == "Dark":
    st.markdown("""
        <style>
        .main {
            background-color: #0e101c;
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #1c1e29;
        }
        .stButton>button {
            background: linear-gradient(to right, #141e30, #243b55);
            border: none;
            color: #fff;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            transition-duration: 0.4s;
        }
        .stButton>button:hover {
            background-color: #ddd;
            color: black;
        }
        .stTextInput>div>input {
            background-color: #1c1e29;
            color: #ffffff;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    title_color = "#ffffff"  # White text for dark mode
else:
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
            color: #000000;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background: linear-gradient(to right, #f0f2f6, #e6e8eb);
            border: none;
            color: #000;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            transition-duration: 0.4s;
        }
        .stButton>button:hover {
            background-color: #333;
            color: white;
        }
        .stTextInput>div>input {
            background-color: #ffffff;
            color: #000000;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    title_color = "#000000"  # Black text for light mode

# Main content
st.markdown(f"<h1 style='color: {title_color};'>🌾 Crop Classifier 🌾</h1>", unsafe_allow_html=True)
st.write("Upload an image to classify the crop type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    prepared_image = prepare_image(image)
    interpreter.set_tensor(input_details[0]['index'], prepared_image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_class = class_labels.get(predicted_class_index, "Unknown")

    st.write(f"🚀 The predicted class of crop is: **{predicted_class}**")

# Sidebar
st.sidebar.title("🌟 About the Project")
st.sidebar.write("""
This project uses a machine learning model to classify images of crops into four categories: **Wheat**, **Maize**, **Cotton**, and **Gram**.

Created by **Pranav Lejith (Amphibiar)**.
                 
Created for AI Project.
""")

st.sidebar.title("💡 Note")
st.sidebar.write("""
This model is still in development and may not always be accurate. 

For the best results, please ensure the wheat images include the stem to avoid confusion with maize.
""")

st.sidebar.title("🛠️ Functionality")
st.sidebar.write("""
This AI model works by using a convolutional neural network (CNN) to analyze images of crops. 
The model has been trained on labeled images of Wheat, Maize, Cotton, and Gram to learn the distinctive features of each crop. 
When you upload an image, the model processes it and predicts the crop type based on the learned patterns.
""")
