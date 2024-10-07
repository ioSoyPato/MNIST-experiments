# MNIST Classification with Multiple Machine Learning Models and Prefect Orchestration

## 🧠 Project Overview
This project leverages Prefect, a powerful workflow orchestration tool, to automate the training and evaluation of multiple machine learning models on the classic MNIST dataset. We are using a variety of models—from classical ML algorithms to deep learning techniques—to classify handwritten digits effectively.

The main goal of this project is to create a flexible and orchestrated workflow for comparing different classification models in terms of accuracy and F1 score, and to provide a simple but powerful baseline for machine learning workflows.

## 🚀 Features
- End-to-end orchestration: Use Prefect to orchestrate data loading, model training, evaluation, and results saving.
- Multiple Models: We evaluate several machine learning and deep learning models:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Random Forest Classifier
    - Convolutional Neural Network (CNN)
    - Multi-Layer Perceptron (MLP)

- Results Logging: The project saves accuracy and F1 score metrics for each model with different hyperparameters into a CSV file, making it easy to analyze the performance.

## 📊 Dataset
The project uses the MNIST dataset, which contains images of handwritten digits from 0 to 9. It is one of the most well-known datasets in the field of machine learning and deep learning. The dataset is loaded directly from TensorFlow's Keras module.

## 🛠️ Technologies Used
- Python: The programming language used for this project.
- Prefect: For orchestrating the workflow, ensuring tasks run in the correct order and handling dependencies.
- TensorFlow/Keras: For building and training deep learning models (CNN).
- Scikit-Learn: For implementing traditional machine learning models like Logistic Regression, SVM, KNN, Random Forest, and MLP.
- Numpy & Pandas: For data manipulation and results storage.
- MLFlow: Optional integration for experiment tracking and visualization of model performance (https://dagshub.com/ioSoyPato/MNIST-experiments).

## ⚙️ Workflow Breakdown
Data Loading (Read Data):
- Loads the MNIST dataset and preprocesses it for both classical ML models and deep learning.

Model Training:

- Each model (Logistic Regression, KNN, SVM, Random Forest, CNN, MLP) is trained with different hyperparameters, allowing you to compare their performance effectively.

Results Saving (Save Results):

- Stores all model evaluation metrics (accuracy and F1 score) into a CSV file for easy analysis.

## 🧩 Models and Hyperparameters
- __Logistic Regression__: Trained with max_iter values of [100, 500, 1000].

- __K-Nearest Neighbors (KNN)__: Trained with n_neighbors values of [3, 5, 7].
- __Support Vector Machine (SVM)__: Trained with regularization parameter C values of [1, 10, 100].
- __Random Forest__: Trained with n_estimators values of [50, 100, 200].
Convolutional Neural Network (CNN): Trained with epochs values of [5, 10, 15].
- __Multi-Layer Perceptron (MLP)__: Trained with different hidden layer configurations: [(64, 32), (128, 64), (256, 128)].

## 📈 Results MNIST
- The results are saved in model_results.csv, which includes:
- Model name
- Hyperparameters used
- Accuracy score
- F1 Score

```csv
Model,Parameter,Accuracy,F1 Score
Logistic Regression,max_iter=100,0.9258,0.9256637959209324
Logistic Regression,max_iter=500,0.9262,0.9260362665058505
Logistic Regression,max_iter=1000,0.9262,0.9260362665058505
KNN,n_neighbors=3,0.9705,0.9704523390961245
KNN,n_neighbors=5,0.9688,0.9687470572168784
KNN,n_neighbors=7,0.9694,0.9693569043443881
SVM,C=1,0.9792,0.9791856837674859
SVM,C=10,0.9837,0.9836954484859449
SVM,C=100,0.9833,0.9832926228555976
Random Forest,n_estimators=50,0.9666,0.966567334555742
Random Forest,n_estimators=100,0.9684,0.9683762773802511
Random Forest,n_estimators=200,0.971,0.9709851218151023
CNN,epochs=5,0.9894999861717224,
CNN,epochs=10,0.9915000200271606,
CNN,epochs=15,0.9908999800682068,
MLP,"hidden_layers=(64, 32)",0.9747,0.9746985925543933
MLP,"hidden_layers=(128, 64)",0.9785,0.9784939791050172
MLP,"hidden_layers=(256, 128)",0.9808,0.9808010536457669
```
Use this CSV file to analyze which model and configuration work best for classifying handwritten digits from the MNIST dataset. In this case the `CNN` with `epochs=15` was `the best model`    

📝 Output
| Model               | Parameter            | Accuracy | F1 Score          |
|---------------------|----------------------|----------|-------------------|
| Logistic Regression | max_iter=100         | 0.9258   | 0.9256637959209324|
| Logistic Regression | max_iter=500         | 0.9262   | 0.9260362665058505|
| Logistic Regression | max_iter=1000        | 0.9262   | 0.9260362665058505|
| KNN                 | n_neighbors=3        | 0.9705   | 0.9704523390961245|
| KNN                 | n_neighbors=5        | 0.9688   | 0.9687470572168784|
| KNN                 | n_neighbors=7        | 0.9694   | 0.9693569043443881|
| SVM                 | C=1                  | 0.9792   | 0.9791856837674859|
| SVM                 | C=10                 | 0.9837   | 0.9836954484859449|
| SVM                 | C=100                | 0.9833   | 0.9832926228555976|
| Random Forest       | n_estimators=50      | 0.9666   | 0.966567334555742 |
| Random Forest       | n_estimators=100     | 0.9684   | 0.9683762773802511|
| Random Forest       | n_estimators=200     | 0.971    | 0.9709851218151023|
| CNN                 | epochs=5             | 0.9895   |                   |
| CNN                 | epochs=10            | 0.9915   |                   |
| `CNN`                 | `epochs=15`            | `0.9909`   |                   |
| MLP                 | hidden_layers=(64, 32)  | 0.9747   | 0.9746985925543933|
| MLP                 | hidden_layers=(128, 64) | 0.9785   | 0.9784939791050172|
| MLP                 | hidden_layers=(256, 128)| 0.9808   | 0.9808010536457669|

## 📈 Results FASHION-MNIST
- The results are saved in model_results.csv, which includes:
- Model name
- Hyperparameters used
- Accuracy score
- F1 Score

```csv
Model,Parameter,Accuracy,F1 Score
Logistic Regression,max_iter=100,0.8439,0.8431108535561094
Logistic Regression,max_iter=500,0.8428,0.8421342996330824
Logistic Regression,max_iter=1000,0.8436,0.8427894400051377
KNN,n_neighbors=3,0.8541,0.8539002124666113
KNN,n_neighbors=5,0.8554,0.8546439722018906
KNN,n_neighbors=7,0.854,0.8534427202406818
SVM,C=1,0.8828,0.8822648793630384
SVM,C=10,0.9002,0.9000975523867072
SVM,C=100,0.8963,0.8960541321771265
Random Forest,n_estimators=50,0.8737,0.8725613322658357
Random Forest,n_estimators=100,0.8775,0.8762058223088482
Random Forest,n_estimators=200,0.8771,0.8757589375940783
CNN,epochs=5,0.8901000022888184,
CNN,epochs=10,0.9041000008583069,
CNN,epochs=15,0.9108999967575073,
MLP,"hidden_layers=(64, 32)",0.8757,0.8750115101384232
MLP,"hidden_layers=(128, 64)",0.8808,0.8801993829827099
MLP,"hidden_layers=(256, 128)",0.8914,0.8912865441859703
```

## 🛠️ Customization

Feel free to modify the models or add your own custom models. You can:

- Add More Hyperparameters: Try experimenting with additional hyperparameters to see how they affect the model performance.
- Add More Models: Include other machine learning or deep learning models to broaden the comparison.

## 🤖 GPU Support
This project is designed to run on CPU by default (`tf.config.set_visible_devices([], 'GPU')`). If you want to enable GPU acceleration for the deep learning models, you can remove or modify that line to use your GPU.

## 🧪 Experiment Tracking with MLFlow
The project is already set up to work with MLFlow for experiment tracking. You can easily modify the code to log metrics and visualize the model performance over time, which is especially useful for larger experiments.

## 🤝 Contributing
Contributions are welcome! If you have ideas for improvements or want to add more models or features, feel free to submit a pull request or open an issue.

## 📄 License
This project is licensed under the __MIT License__. See the LICENSE file for more details.

## 📧 Contact
If you have any questions or suggestions, feel free to reach out at patricio.villanueva.gio.ICD@outlook.com.