import pandas as pd
import joblib
import os

# Load the saved model
try:
    os.chdir(os.path.dirname(__file__))
except:
    pass
best_model = joblib.load('best_car_price_model.pkl')


# Collect user input for new data with default values
print("Enter the features for prediction (press Enter to use default values):")
wheelbase = float(input("Wheelbase (default: 100): ") or 100)
carlength = float(input("Car Length (default: 180): ") or 180)
carwidth = float(input("Car Width (default: 70): ") or 70)
carheight = float(input("Car Height (default: 50): ") or 50)
curbweight = float(input("Curb Weight (default: 3000): ") or 3000)
enginesize = float(input("Engine Size (default: 150): ") or 150)
boreratio = float(input("Bore Ratio (default: 3): ") or 3)
stroke = float(input("Stroke (default: 3): ") or 3)
compressionratio = float(input("Compression Ratio (default: 9): ") or 9)
horsepower = float(input("Horsepower (default: 120): ") or 120)
peakrpm = float(input("Peak RPM (default: 5000): ") or 5000)
citympg = float(input("City MPG (default: 25): ") or 25)
highwaympg = float(input("Highway MPG (default: 30): ") or 30)
fueltype = input("Fuel Type (default: Gas): ").lower() or 'gas'
aspiration = input("Aspiration (default: Std): ").lower() or 'std'
doornumber = input("Door Number (default: Four): ").lower() or 'four'
carbody = input("Car Body (default: Sedan): ").lower() or 'sedan'
drivewheel = input("Drive Wheel (default: FWD): ").lower() or 'fwd'
enginelocation = input("Engine Location (default: Front): ").lower() or 'front'
enginetype = input("Engine Type (default: OHC): ").lower() or 'ohc'
cylindernumber = input("Cylinder Number (default: Four): ").lower() or 'four'
fuelsystem = input("Fuel System (default: MPFI): ").lower() or 'mpfi'

# Create a DataFrame from the user input
new_data = pd.DataFrame({
    'wheelbase': [wheelbase],
    'carlength': [carlength],
    'carwidth': [carwidth],
    'carheight': [carheight],
    'curbweight': [curbweight],
    'enginesize': [enginesize],
    'boreratio': [boreratio],
    'stroke': [stroke],
    'compressionratio': [compressionratio],
    'horsepower': [horsepower],
    'peakrpm': [peakrpm],
    'citympg': [citympg],
    'highwaympg': [highwaympg],
    'fueltype': [fueltype],
    'aspiration': [aspiration],
    'doornumber': [doornumber],
    'carbody': [carbody],
    'drivewheel': [drivewheel],
    'enginelocation': [enginelocation],
    'enginetype': [enginetype],
    'cylindernumber': [cylindernumber],
    'fuelsystem': [fuelsystem]
})

# Make predictions
predictions = best_model.predict(new_data)

# Display the predictions
print("\nPredicted Car Price:")
print(predictions)
