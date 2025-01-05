import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Probability of Default (PD) - Logistic Regression
class CreditRiskModel:
    def _init_(self):
        self.model = LogisticRegression()

    def train_model(self, X, y):
        """Train the logistic regression model"""
        self.model.fit(X, y)

    def predict_pd(self, X):
        """Predict probability of default"""
        return self.model.predict_proba(X)[:, 1]

# Loss Given Default (LGD) Calculation
def calculate_lgd(recovery_value, ead):
    """Calculate Loss Given Default (LGD)"""
    return 1 - (recovery_value / ead)

# Expected Loss (EL) Calculation
def calculate_expected_loss(pd, lgd, ead):
    """Calculate Expected Loss (EL)"""
    return pd * lgd * ead

# Example Pipeline
if _name_ == "_main_":
    # Example Data (Replace with actual dataset)
    data = {
        "income": [50000, 60000, 30000, 40000, 70000],
        "debt": [10000, 15000, 20000, 12000, 30000],
        "credit_score": [700, 650, 600, 680, 750],
        "default": [0, 1, 1, 0, 0],  # Target variable
        "recovery_value": [8000, 10000, 15000, 9000, 20000],
        "ead": [12000, 15000, 20000, 11000, 30000],
    }

    df = pd.DataFrame(data)

    # Features and Target
    X = df[["income", "debt", "credit_score"]]
    y = df["default"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression for PD
    model = CreditRiskModel()
    model.train_model(X_train, y_train)

    # Predict PD for test set
    predicted_pd = model.predict_pd(X_test)
    print("Predicted PD:", predicted_pd)

    # Calculate LGD for each observation
    df["lgd"] = calculate_lgd(df["recovery_value"], df["ead"])
    print("LGD Values:", df["lgd"].values)

    # Calculate Expected Loss for test set
    ead_test = df.loc[X_test.index, "ead"]
    lgd_test = df.loc[X_test.index, "lgd"]

    expected_loss = calculate_expected_loss(predicted_pd, lgd_test, ead_test)
    print("Expected Loss:", expected_loss)

