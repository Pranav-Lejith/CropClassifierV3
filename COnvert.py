import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('crop_classifier_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('crop_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite format and saved as 'crop_classifier_model.tflite'")
