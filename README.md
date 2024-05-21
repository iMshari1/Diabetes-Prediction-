# Diabetes Prediction and Performance Evaluation Using Neural Networks

## Project Overview

This project aims to predict diabetes and evaluate the performance of the prediction model using neural networks. The project uses a dataset containing various health metrics to train a neural network model that classifies whether a person has diabetes. The model is evaluated using accuracy and confusion matrix metrics.

## Dataset

The dataset used for this project is stored in an Excel file named `diabetes.data.xls`. It contains various health-related features used to predict diabetes. The dataset includes the following columns:
- Feature columns: 8 health metrics/features.
- Target column: Binary classification indicating the presence of diabetes.

## Requirements

- Python 3.x
- Pandas
- NumPy
- TensorFlow (Keras)
- Scikit-learn
- Matplotlib
- Seaborn
- Openpyxl (for reading Excel files)

## Installation

1. Clone the repository or download the project files:
    ```bash
    git clone https://github.com/yourusername/DiabetesPrediction.git
    cd DiabetesPrediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install pandas numpy tensorflow scikit-learn matplotlib seaborn openpyxl
    ```

## Project Files

- `diabetes.data.xls`: The dataset containing health metrics and diabetes classification.
- `diabetes_prediction.py`: The main script to run the diabetes prediction and model evaluation.

## Steps

1. **Data Loading and Preprocessing:**
    - Load the dataset from the Excel file.
    - Separate input features and output target variable.

2. **Model Definition and Training:**
    - Define a neural network model using TensorFlow and Keras.
    - Compile the model with binary cross-entropy loss and Adam optimizer.
    - Train the model on the dataset.

3. **Model Evaluation:**
    - Evaluate the model using accuracy metrics.
    - Generate a confusion matrix to visualize the classification performance.
    - Plot the confusion matrix using Matplotlib and Seaborn.

4. **Prediction and Comparison:**
    - Make predictions on the input data.
    - Compare actual and predicted values.
