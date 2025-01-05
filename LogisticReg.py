import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Generate Synthetic Data
def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    data = {
        "Income": np.random.randint(30, 120, num_samples),  # Income in thousands
        "DTI": np.random.uniform(10, 50, num_samples),  # Debt-to-Income ratio in %
        "Credit_History": np.random.uniform(50, 100, num_samples),  # Score (0-100)
        "Spending_Behavior": np.random.uniform(30, 80, num_samples),  # Score (0-100)
        "Age": np.random.randint(21, 70, num_samples),  # Age in years
        "Employment_Stability": np.random.uniform(40, 100, num_samples),  # Score (0-100)
    }
    df = pd.DataFrame(data)
    
    # Generate Credit Score and Classify it into Categories
    df["Credit_Score"] = np.random.uniform(300, 850, num_samples)
    df["Category"] = pd.cut(
        df["Credit_Score"],
        bins=[300, 579, 669, 739, 850],
        labels=["Poor", "Fair", "Good", "Excellent"]
    )
    return df

# Step 2: Prepare the data
data = generate_synthetic_data(100)
X = data[["Income", "DTI", "Credit_History", "Spending_Behavior", "Age", "Employment_Stability"]]
y = data["Category"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Predict categories for new individuals
def predict_credit_category(new_data):
    new_data = scaler.transform(new_data)  # Scale the new data
    return model.predict(new_data)

# Example: Predict for 3 new individuals
new_individuals = pd.DataFrame({
    "Income": [50, 85, 95],  # in thousands
    "DTI": [20, 35, 25],  # in %
    "Credit_History": [75, 90, 85],  # in 0-100 scale
    "Spending_Behavior": [60, 45, 70],  # in 0-100 scale
    "Age": [30, 40, 50],  # in years
    "Employment_Stability": [80, 90, 95]  # in 0-100 scale
})

predicted_categories = predict_credit_category(new_individuals)
print(f"Predicted Credit Categories: {predicted_categories}")
