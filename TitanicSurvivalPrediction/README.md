# Titanic Survival Prediction Model

This project uses the Titanic dataset to predict whether a passenger would survive or not. The model is trained using the XGBoost algorithm.

## Overview

The project consists of two main Python scripts:

1. `model.py`: This script is responsible for training the XGBoost model. It includes data preprocessing, model training, prediction, and evaluation steps. It also saves the trained model to a file.

2. `prediction.py`: This script loads the trained XGBoost model and uses it to make a prediction on user input data. The user input data is collected through the console and includes features such as passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, fare, and embarked port.

## Data Preprocessing

The data preprocessing steps include:

- Loading the Titanic data from a CSV file.
- Dropping unnecessary columns including 'PassengerId', 'Ticket', 'Cabin', and 'Name'.
- Filling missing values in 'Age', 'Embarked', and 'Fare'.
- Converting categorical data to numeric using label encoding.
- LabelEncoder is used to convert the categorical features ‘Sex’ and ‘Embarked’ into numerical values. This is necessary because the XGBoost model, which is being used for prediction, can only handle numerical values.It's working can be summarized as this 
           - Instantiating the LabelEncoder.
           - Fit and Transform

## Model Training

The model is trained using the XGBoost algorithm with the following parameters:

- Objective: 'binary:logistic'
- Evaluation metric: 'logloss'
- Learning rate (eta): 0.1
- Maximum depth of trees: 3
- Subsample ratio: 0.8
- Column sample by tree: 0.8
- Random seed: 42

The model is trained for 100 rounds with early stopping if the performance doesn't improve after 10 rounds.

## Model Evaluation

The model's performance is evaluated using accuracy score. The model's predictions are binary (1 for survived, 0 for not survived).

## Model Prediction

The prediction script collects user input data, converts the categorical data to numeric using label encoding, makes the prediction, and displays the prediction result.

## Note

The current model tends to overfit the training data. Future work could explore techniques to reduce overfitting, such as regularization, more extensive hyperparameter tuning, or using a different model.

## Dependencies

- pandas
- xgboost
- sklearn
