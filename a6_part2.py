"""
Assignment 6 Part 2: House Price Prediction (Multivariable Regression)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the house price data and explore it
    """
    data = pd.read_csv(filename)
    
    print("=== House Price Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print(f"\nBasic statistics:")
    print(data.describe())
    
    print(f"\nColumn names: {list(data.columns)}")
    
    return data


def visualize_features(data):
    """
    Create scatter plots for each feature vs Price
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('House Features vs Price', fontsize=16, fontweight='bold')
    
    # Plot 1: SquareFeet vs Price
    axes[0, 0].scatter(data['SquareFeet'], data['Price'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Square Feet')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('SquareFeet vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Bedrooms vs Price
    axes[0, 1].scatter(data['Bedrooms'], data['Price'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Bedrooms vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Bathrooms vs Price
    axes[1, 0].scatter(data['Bathrooms'], data['Price'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Bathrooms')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Bathrooms vs Price')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Age vs Price
    axes[1, 1].scatter(data['Age'], data['Price'], color='orange', alpha=0.6)
    axes[1, 1].set_xlabel('Age (years)')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].set_title('Age vs Price')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'feature_plots.png'")
    plt.show()


def prepare_features(data):
    """
    Separate features (X) from target (y)
    """
    feature_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    X = data[feature_columns]
    y = data['Price']
    
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    
    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train a multivariable linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    # Print full equation
    equation = "Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(f"\nEquation:\n{equation}")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model's performance
    """
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}  → Explains {r2*100:.2f}% of variance")
    print(f"Root Mean Squared Error: ${rmse:.2f}  → Avg prediction error")
    
    # Feature importance
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    """
    Show side-by-side comparison of actual vs predicted prices
    """
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    
    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100
        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")


def make_prediction(model, sqft, bedrooms, bathrooms, age):
    """
    Make a prediction for a specific house
    """
    house_features = pd.DataFrame([[sqft, bedrooms, bathrooms, age]],
                                  columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age'])
    predicted_price = model.predict(house_features)[0]
    
    print(f"\n=== New House Prediction ===")
    print(f"Specs: {sqft} sqft, {bedrooms} bedrooms, {bathrooms} bathrooms, {age} years old")
    print(f"Predicted price: ${predicted_price:,.2f}")
    
    return predicted_price


if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - MULTIVARIABLE LINEAR REGRESSION")
    print("=" * 70)
    
    # Step 1: Load and explore
    data = load_and_explore_data('house_prices.csv')
    
    # Step 2: Visualize all features
    visualize_features(data)
    
    # Step 3: Prepare features
    X, y = prepare_features(data)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Train model
    model = train_model(X_train, y_train, X.columns)
    
    # Step 6: Evaluate model
    predictions = evaluate_model(model, X_test, y_test, X.columns)
    
    # Step 7: Compare predictions
    compare_predictions(y_test, predictions, num_examples=10)
    
    # Step 8: Make a new prediction
    make_prediction(model, 2500, 4, 3, 10)  # Example house
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("=" * 70)
