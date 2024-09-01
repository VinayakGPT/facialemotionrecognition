import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('trained_model.h5') 

# List of class labels
class_labels = [
    'surprise',
    'fear',
    'angry',
    'neutral',
    'sad',
    'disgust',
    'happy'
]

st.title('Emotion Classification with Streamlit')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, color_mode='grayscale', target_size=(48, 48)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])  
    predicted_class_label = class_labels[predicted_class_index] 

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Prediction:")
    st.write(predicted_class_label) 
