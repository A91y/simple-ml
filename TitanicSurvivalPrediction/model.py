import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Titanic data from CSV
df = pd.read_csv('TitanicData.csv')

# Drop unnecessary columns including 'Cabin'
columns_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
df = df.drop(columns=columns_to_drop)

# Fill missing values in 'Age', 'Embarked', 'Fare'
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Convert categorical data to numeric using label encoding
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Split the data into features (X) and target variable (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds, evals=[(dtest, 'eval')], early_stopping_rounds=10)

# Make predictions on the test set
y_pred = model.predict(dtest)

# Convert probabilities to binary predictions
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save_model('titanic_model_xgboost.json')

## This model is overfitted!