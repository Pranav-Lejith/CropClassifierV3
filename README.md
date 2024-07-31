# CropIdentifierV2.6.0

This project was created by Pranav Lejith on 24th July 2024

# Function

Differentiates Wheat and Maize using image classification model built using Tensorflow

# Requirements

Need streamlit and tensorflow and some other libriaries
Download the model using this link
https://drive.google.com/file/d/18HQqSdCpw0hde0TQlBPWuXgxZN711_Ir/view?usp=sharing  (for .h5)
https://drive.google.com/file/d/1mbIE1DzvU2VGVij8ETMXuRDtZ9JXuw4Y/view?usp=sharing  (for .tflite)

Install streamlit and tensorflow using
```ruby
pip install streamlit
pip install tensorflow
```
# Executing the code(If you did not download the model)

To run the project if you have not downloaded the model, run the trainModel.py file using 
```ruby
python trainModel.py
```
The file crop_classifier_model.h5 file would have been generated
Now Run
```ruby
python Convert.py
```
This converts the .h5 file extenstion into a .tflite extension.

Next, to run the script, type
```ruby
streamlit run ModelInvoker.py
```
# Executing the code(If you have the model downloaded)
To run the project if you have the model by either downloading it or by creating it using trainModel.py , type
```ruby
streamlit run ModelInvoker.py
```

# Credits
Amphibiar
