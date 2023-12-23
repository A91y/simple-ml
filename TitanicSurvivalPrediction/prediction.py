import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved XGBoost model
model = xgb.Booster()
model.load_model('titanic_model_xgboost.json')

# Function to get user input for prediction
def get_user_input():
    pclass = int(input("Enter passenger class (1, 2, or 3): "))
    sex = input("Enter sex (male or female): ").lower()
    age = float(input("Enter age: "))
    sibsp = int(input("Enter number of siblings/spouses aboard: "))
    parch = int(input("Enter number of parents/children aboard: "))
    fare = float(input("Enter fare: "))
    embarked = input("Enter embarked port (C, Q, or S): ").upper()

    user_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }

    return pd.DataFrame([user_data])

# Get user input
user_input_data = get_user_input()

# Convert categorical data to numeric using label encoding
label_encoder = LabelEncoder()
user_input_data['Sex'] = label_encoder.fit_transform(user_input_data['Sex'])
user_input_data['Embarked'] = label_encoder.fit_transform(user_input_data['Embarked'])

# Make prediction
user_input_dmatrix = xgb.DMatrix(user_input_data)
prediction = model.predict(user_input_dmatrix)[0]

# Convert probability to binary prediction
prediction_binary = 1 if prediction > 0.5 else 0

# Display the prediction
if prediction_binary == 1:
    print("The passenger is predicted to survive by chance of " + str(prediction * 100) + "%.")
else:
    print("The passenger is predicted not to survive by chance of" + str((1 - prediction) * 100) + "%.")
