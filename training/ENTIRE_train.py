import prefect
from prefect import task, flow
import numpy as np
import pandas as pd
import mlflow
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

@task(name="Read Data MNIST")
def readData():
    # Cargar datos MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Aplanar imágenes para modelos clásicos de ML
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Normalizar datos para redes neuronales
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    return x_train, x_test, x_train_flat, x_test_flat, y_train, y_test

@task(name="Save Results MNIST")
def save_results(results):
    # Crear un DataFrame a partir de los resultados
    df = pd.DataFrame(results)
    
    # Guardar los resultados en un archivo CSV
    df.to_csv('../results/MNIST_model_results.csv', index=False)

@task(name="Logistic Regression MNIST")
def logisticRegression():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()

    # Parámetros para Regresión Logística
    param_list = [100, 500, 1000]
    results = []

    for max_iter in param_list:
        log_reg = LogisticRegression(max_iter=max_iter)
        log_reg.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_log_reg = log_reg.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_log_reg)
        f1 = f1_score(y_test, y_pred_log_reg, average='weighted')
        
        results.append({
            'Model': 'Logistic Regression',
            'Parameter': f'max_iter={max_iter}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="KNN MNIST")
def KNN():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()
    
    # Parámetros para KNN
    param_list = [3, 5, 7]
    results = []

    for n_neighbors in param_list:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_knn = knn.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_knn)
        f1 = f1_score(y_test, y_pred_knn, average='weighted')
        
        results.append({
            'Model': 'KNN',
            'Parameter': f'n_neighbors={n_neighbors}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="SVM MNIST")
def SVM():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()

    # Parámetros para SVM
    param_list = [1, 10, 100]
    results = []

    for C in param_list:
        svm = SVC(kernel='rbf', C=C)
        svm.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_svm = svm.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_svm)
        f1 = f1_score(y_test, y_pred_svm, average='weighted')

        results.append({
            'Model': 'SVM',
            'Parameter': f'C={C}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="Random Forest MNIST")
def randomForest():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()

    # Parámetros para Random Forest
    param_list = [50, 100, 200]
    results = []

    for n_estimators in param_list:
        random_forest = RandomForestClassifier(n_estimators=n_estimators)
        random_forest.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_rf = random_forest.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_rf)
        f1 = f1_score(y_test, y_pred_rf, average='weighted')

        results.append({
            'Model': 'Random Forest',
            'Parameter': f'n_estimators={n_estimators}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="CNN MNIST")
def CNN():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()

    # Parámetros para CNN
    param_list = [5, 10, 15]
    results = []

    for epochs in param_list:
        # Red Neuronal Convolucional (CNN)
        cnn_model = keras.models.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compilar el modelo CNN

        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo CNN
        cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2)

        # Evaluación
        cnn_score = cnn_model.evaluate(x_test, y_test, verbose=0)
        acc = cnn_score[1]

        results.append({
            'Model': 'CNN',
            'Parameter': f'epochs={epochs}',
            'Accuracy': acc,
            'F1 Score': None  # F1 Score no se calcula explícitamente en este caso
        })

    return results

@task(name="MLP sgd MNIST")
def MLP():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()

    # Parámetros para MLP
    param_list = [(64, 32), (128, 64), (256, 128)]
    results = []

    for hidden_layers in param_list:
        mlp_variant = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=300, solver='sgd', learning_rate_init=0.01)
        mlp_variant.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_mlp_variant = mlp_variant.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_mlp_variant)
        f1 = f1_score(y_test, y_pred_mlp_variant, average='weighted')

        results.append({
            'Model': 'MLP',
            'Parameter': f'hidden_layers={hidden_layers}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results


@task(name="Save best model MNIST")
def saveBestModel():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readData()
    cnn_model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.2)
    cnn_model.save("../models/cnn_digit_model.h5")

    return cnn_model

@task(name="MainFlow MNIST")
def MainFlow():
    results = []

    # Train using Logistic Regression
    results.extend(logisticRegression())
    # Train using KNN
    results.extend(KNN())
    # Train using SVM
    results.extend(SVM())
    # Train using Random Forest
    results.extend(randomForest())
    # Train using CNN
    results.extend(CNN())
    # Train using MLP
    results.extend(MLP())

    # Save results to CSV
    save_results(results)

    # Save best model
    saveBestModel()


@task(name="Read Data FASHION")
def readDataFashion():
    # Cargar datos Fashion MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Aplanar imágenes para modelos clásicos de ML
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Normalizar datos para redes neuronales (escalar valores a [0, 1] y añadir dimensión de canal)
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    return x_train, x_test, x_train_flat, x_test_flat, y_train, y_test


@task(name="Save Results FASHION")
def save_resultsFashion(results):
    # Crear un DataFrame a partir de los resultados
    df = pd.DataFrame(results)
    
    # Guardar los resultados en un archivo CSV
    df.to_csv('../results/FASHIONMNIST_model_results.csv', index=False)

@task(name="Logistic Regression FASHION")
def logisticRegressionFashion():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readDataFashion()

    # Parámetros para Regresión Logística
    param_list = [100, 500, 1000]
    results = []

    for max_iter in param_list:
        log_reg = LogisticRegression(max_iter=max_iter)
        log_reg.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_log_reg = log_reg.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_log_reg)
        f1 = f1_score(y_test, y_pred_log_reg, average='weighted')
        
        results.append({
            'Model': 'Logistic Regression',
            'Parameter': f'max_iter={max_iter}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="KNN FASHION")
def KNNFashion():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readDataFashion()
    
    # Parámetros para KNN
    param_list = [3, 5, 7]
    results = []

    for n_neighbors in param_list:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_knn = knn.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_knn)
        f1 = f1_score(y_test, y_pred_knn, average='weighted')
        
        results.append({
            'Model': 'KNN',
            'Parameter': f'n_neighbors={n_neighbors}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="SVM FASHION")
def SVMFashion():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readDataFashion()

    # Parámetros para SVM
    param_list = [1, 10, 100]
    results = []

    for C in param_list:
        svm = SVC(kernel='rbf', C=C)
        svm.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_svm = svm.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_svm)
        f1 = f1_score(y_test, y_pred_svm, average='weighted')

        results.append({
            'Model': 'SVM',
            'Parameter': f'C={C}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="Random Forest FASHION")
def randomForestFashion():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readDataFashion()

    # Parámetros para Random Forest
    param_list = [50, 100, 200]
    results = []

    for n_estimators in param_list:
        random_forest = RandomForestClassifier(n_estimators=n_estimators)
        random_forest.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_rf = random_forest.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_rf)
        f1 = f1_score(y_test, y_pred_rf, average='weighted')

        results.append({
            'Model': 'Random Forest',
            'Parameter': f'n_estimators={n_estimators}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="CNN FASHION")
def CNNFashion():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readDataFashion()

    # Parámetros para CNN
    param_list = [5, 10, 15]
    results = []

    for epochs in param_list:
        # Red Neuronal Convolucional (CNN)
        cnn_model = keras.models.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compilar el modelo CNN

        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo CNN
        cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2)

        # Evaluación
        cnn_score = cnn_model.evaluate(x_test, y_test, verbose=0)
        acc = cnn_score[1]

        results.append({
            'Model': 'CNN',
            'Parameter': f'epochs={epochs}',
            'Accuracy': acc,
            'F1 Score': None  # F1 Score no se calcula explícitamente en este caso
        })

    return results

@task(name="MLP sgd FASHION")
def MLPFashion():
    x_train, x_test, x_train_flat, x_test_flat, y_train, y_test = readDataFashion()

    # Parámetros para MLP
    param_list = [(64, 32), (128, 64), (256, 128)]
    results = []

    for hidden_layers in param_list:
        mlp_variant = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=300, solver='sgd', learning_rate_init=0.01)
        mlp_variant.fit(x_train_flat, y_train)

        # Evaluación
        y_pred_mlp_variant = mlp_variant.predict(x_test_flat)
        acc = accuracy_score(y_test, y_pred_mlp_variant)
        f1 = f1_score(y_test, y_pred_mlp_variant, average='weighted')

        results.append({
            'Model': 'MLP',
            'Parameter': f'hidden_layers={hidden_layers}',
            'Accuracy': acc,
            'F1 Score': f1
        })

    return results

@task(name="MainFlow FASHION")
def MainFlowFashion():
    results = []

    # Train using Logistic Regression
    results.extend(logisticRegressionFashion())
    # Train using KNN
    results.extend(KNNFashion())
    # Train using SVM
    results.extend(SVMFashion())
    # Train using Random Forest
    results.extend(randomForestFashion())
    # Train using CNN
    results.extend(CNNFashion())
    # Train using MLP
    results.extend(MLPFashion())

    # Save results to CSV
    save_results(results)

@flow(name="TRAINING")
def training():
    MainFlow()
    MainFlowFashion()

if __name__ == "__main__":
    training()
