import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# Load MNIST data from tensorflow datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten the images for Logistic Regression, KNN, SVM, and Random Forest models
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

# Normalize the data for neural network models
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0


# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train_flat, y_train)

# Evaluate performance
y_pred_log_reg = log_reg.predict(x_test_flat)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))


# K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_flat, y_train)

# Evaluate performance
y_pred_knn = knn.predict(x_test_flat)
print("K-Nearest Neighbors:")
print(classification_report(y_test, y_pred_knn))


# Support Vector Machine model
svm = SVC(kernel='rbf', C=10)
svm.fit(x_train_flat, y_train)

# Evaluate performance
y_pred_svm = svm.predict(x_test_flat)
print("SVM:")
print(classification_report(y_test, y_pred_svm))


# Random Forest model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train_flat, y_train)

# Evaluate performance
y_pred_rf = random_forest.predict(x_test_flat)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))


# MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, solver='adam')
mlp.fit(x_train_flat, y_train)

# Evaluate performance
y_pred_mlp = mlp.predict(x_test_flat)
print("Neural Network (MLP):")
print(classification_report(y_test, y_pred_mlp))


# CNN Model
cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate performance
cnn_score = cnn_model.evaluate(x_test, y_test, verbose=0)
print(f"CNN Test Loss: {cnn_score[0]}, Test Accuracy: {cnn_score[1]}")


# MLP with different architecture and solver
mlp_variant = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, solver='sgd', learning_rate_init=0.01)
mlp_variant.fit(x_train_flat, y_train)

# Evaluate performance
y_pred_mlp_variant = mlp_variant.predict(x_test_flat)
print("Neural Network (MLP) Variant:")
print(classification_report(y_test, y_pred_mlp_variant))

