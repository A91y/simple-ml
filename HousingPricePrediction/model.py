# Import necessary libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv("Housing.csv")

# Data preprocessing
# Handle missing values, convert categorical variables, split into X and y
# X represents the features, and y represents the target variable (price)
X = data.drop("price", axis=1)
y = data["price"]

# Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling and encoding (if needed)
# Define numeric and categorical features for preprocessing
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Create transformers for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Create a preprocessor using ColumnTransformer
# Apply transformations to numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train multiple models with hyperparameter tuning
# Models include Linear Regression, Decision Tree, and Random Forest
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': GridSearchCV(DecisionTreeRegressor(), {'max_depth': [None, 10, 20, 30]}, cv=5, scoring='neg_mean_squared_error'),
    'Random Forest': GridSearchCV(RandomForestRegressor(), {'n_estimators': [50, 100, 200]}, cv=5, scoring='neg_mean_squared_error')
}

# Variables to store the best model information
best_model_name = None
best_model = None
best_mse = float('inf')

# Iterate over models and evaluate their performance
for name, model in models.items():
    # Create a pipeline with preprocessing and the current model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse}")
    
    # Save the best model based on the lowest MSE
    if mse < best_mse:
        best_mse = mse
        best_model_name = name
        best_model = pipeline

# Choose the best model based on performance
print(f"\nBest Model: {best_model_name} - Best Mean Squared Error: {best_mse}")

# Save the best model using joblib
# This model can be loaded and used for future predictions
joblib.dump(best_model, 'best_model.pkl')

