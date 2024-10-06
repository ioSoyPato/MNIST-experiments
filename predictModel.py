import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado (solo una vez)
model = load_model("./models/cnn_digit_model.h5")

def predictedImageArray(img_array):
    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class


