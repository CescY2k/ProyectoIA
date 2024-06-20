import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo
@st.cache(allow_output_mutation=True)
def load_my_model():
    return load_model('Flores_Recog_Model.h5')

model = load_my_model()

# Nombres de las flores
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Función para clasificar imágenes
def classify_image(image):
    input_image = tf.image.resize(image, (180, 180))
    input_image = tf.expand_dims(input_image, axis=0)
    predictions = model.predict(input_image)
    score = tf.nn.softmax(predictions[0])
    return flower_names[np.argmax(score)], 100 * np.max(score)

# Interfaz de usuario con Streamlit
st.title('Flower Classification CNN Model')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.image.decode_image(uploaded_file.read(), channels=3)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    class_name, confidence = classify_image(image)
    st.write(f"Prediction: {class_name}, Confidence: {confidence:.2f}%")
