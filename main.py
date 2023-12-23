import subprocess
import sys

def main():
    print("Select the model you want to run:")
    print("1. Car Price Prediction")
    print("2. Housing Price Prediction")
    print("3. Titanic Survival Prediction")
    choice = input()

    if choice == '1':
        model_dir = 'CarPricePrediction'
    elif choice == '2':
        model_dir = 'HousingPricePrediction'
    elif choice == '3':
        model_dir = 'TitanicSurvivalPrediction'
    else:
        print("Invalid choice")
        sys.exit(1)

    print("Select the operation you want to perform:")
    print("1. Train model")
    print("2. Make predictions")
    operation = input()

    if operation == '1':
        script = 'model.py'
    elif operation == '2':
        script = 'prediction.py'
    else:
        print("Invalid operation")
        sys.exit(1)

    subprocess.run([sys.executable, f'{model_dir}/{script}'])

if __name__ == "__main__":
    main()
