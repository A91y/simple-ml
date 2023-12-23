import pandas as pd
import joblib

def preprocess_input(input_data):
    # Implement the same preprocessing steps as in the training phase
    # Adjust this function based on the preprocessing steps used in your training script
    return input_data

def main():
    # Load the saved model
    loaded_model = joblib.load('best_model.pkl')

    # Get user input for house features
    print("Enter the details for the house:")
    area = float(input("Area (in square feet): "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of bathrooms: "))
    stories = int(input("Number of stories: "))
    parking = int(input("Number of parking spaces: "))
    mainroad = input("Main road access (yes/no): ").lower()
    guestroom = input("Guest room (yes/no): ").lower()
    basement = input("Basement (yes/no): ").lower()
    hotwaterheating = input("Hot water heating (yes/no): ").lower()
    airconditioning = input("Air conditioning (yes/no): ").lower()
    prefarea = input("Preferred area (yes/no): ").lower()
    furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ").lower()

    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'parking': [parking],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    # Preprocess the user input
    input_data = preprocess_input(input_data)

    # Make predictions
    prediction = loaded_model.predict(input_data)

    # Display the prediction
    print("\nPredicted house price:")
    print(prediction)

if __name__ == "__main__":
    main()
