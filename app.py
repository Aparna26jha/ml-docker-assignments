# print("Hello from Aparna's Docker container!")
# print("Docker image created and executed successfully.")

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("HOUSE RENT PREDICTION USING LINEAR REGRESSION")
    print("=" * 60)

    # Locate the CSV file relative to this Python file.
    project_directory = Path(__file__).resolve().parent
    dataset_path = project_directory / "rent_data.csv"

    # Load the dataset.
    data = pd.read_csv(dataset_path)

    print("\nDataset loaded successfully.")
    print(f"Total records: {len(data)}")
    print("\nFirst five records:")
    print(data.head())

    # Input features used by the model.
    feature_columns = [
        "area_sqft",
        "bedrooms",
        "age_years",
        "distance_km",
    ]

    X = data[feature_columns]
    y = data["monthly_rent"]

    # Separate the data into training and testing portions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Create and train the Linear Regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel trained successfully.")

    # Test the trained model.
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    print("\nModel evaluation:")
    print(f"Mean Absolute Error: Rs. {mae:,.2f}")
    print(f"R-squared Score: {r2:.4f}")

    # Example house for rent prediction.
    example_house = pd.DataFrame(
        [
            {
                "area_sqft": 1000,
                "bedrooms": 2,
                "age_years": 4,
                "distance_km": 8,
            }
        ]
    )

    predicted_rent = model.predict(example_house)[0]

    print("\nExample property:")
    print("Area: 1000 square feet")
    print("Bedrooms: 2")
    print("Property age: 4 years")
    print("Distance: 8 kilometres")
    print(f"\nPredicted monthly rent: Rs. {predicted_rent:,.2f}")

    print("\nML application executed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()