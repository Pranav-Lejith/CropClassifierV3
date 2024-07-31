import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Function to load the TensorFlow Lite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to get class labels based on the selected model
def get_class_labels(model_selection):
    if model_selection == "Wheat and Maize":
        return {0: 'Wheat', 1: 'Maize'}
    elif model_selection == "Wheat, Maize, Cotton, and Gram":
        return {0: 'Wheat', 1: 'Maize', 2: 'Cotton', 3: 'Gram'}

# Function to prepare the image for the model
def prepare_image(image, model_selection):
    if model_selection == "Wheat and Maize":
        image_size = (150, 150)
    else:
        image_size = (224, 224)

    image = image.resize(image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image.astype(np.float32)

# Sidebar for theme selection
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
    title_color = "#ffffff"
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
    title_color = "#000000"

# Main content
st.markdown(f"<h1 style='color: {title_color};'>🌾 Crop Classifier 🌾</h1>", unsafe_allow_html=True)
st.write("Upload an image to classify the crop type.")

# Model selection
model_selection = st.sidebar.selectbox(
    "Choose the model",
    ("Wheat and Maize", "Wheat, Maize, Cotton, and Gram")
)

if model_selection == "Wheat and Maize":
    model_path = "crop_classifier_model_wheat_maize.tflite"
else:
    model_path = "crop_classifier_model_wheat_maize_cotton_gram.tflite"

# Load the selected model
interpreter = load_model(model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the class labels based on the model
class_labels = get_class_labels(model_selection)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    prepared_image = prepare_image(image, model_selection)
    interpreter.set_tensor(input_details[0]['index'], prepared_image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_class = class_labels.get(predicted_class_index, "Unknown")

    st.write(f"🚀 The predicted class of crop is: **{predicted_class}**")

# Sidebar content
with st.sidebar:
    st.markdown("### ℹ️ Info")
    with st.expander("Model Accuracy Information"):
        st.write("""
        <style>
        .dataframe {
            border-collapse: collapse;
            width: 100%;
        }
        .dataframe th, .dataframe td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .dataframe th {
            background-color: #f2f2f2;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)

        st.write("""
        The following table provides information about the accuracy of the models:

        | Model                               | Accuracy       | Crops |
        |-------------------------------------|----------------|-------|
        | Wheat and Maize                     | Medium         | 2     |
        | Wheat, Maize, Cotton, and Gram      | High           | 4     |
        """, unsafe_allow_html=True)

    st.sidebar.title("🌟 About the Project")
    st.sidebar.write(f"""
    This project uses a machine learning model to classify images of crops into the selected categories.

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
    The model has been trained on labeled images of the selected crops to learn the distinctive features of each crop. 
    When you upload an image, the model processes it and predicts the crop type based on the learned patterns.
    """)
