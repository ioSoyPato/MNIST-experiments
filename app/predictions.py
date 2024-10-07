import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pickle
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your model architecture
def create_model():
    #Modelo conv sin momentum
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) #capa convolucional para imagenes, 
    model.add(MaxPooling2D((2, 2)))#filtros 2D

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax')) # salida softmax

    return model



def predictData(img_path):
    model = create_model()
    with open('./models/CNN_MNIST.pkl', 'rb') as file:
        weights = pickle.load(file)

    model.set_weights(weights)
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.invert(img)
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0

    img_array = img_array.reshape((1, 28, 28, 1))  

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    return int(predicted_class)
