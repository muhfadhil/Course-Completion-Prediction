# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# Load the data
df = pd.read_pickle("../../data/processed/02_data_processed.pkl")

# Modeling
X = df.drop("CourseCompletion", axis=1)
y = df.CourseCompletion

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Modeling
# Make function for training model and evaluate model
def train_model(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1):
    # Define GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Return best parameters
    return grid_search


# Function to evaluate metrics
def evaluate_metrics(model, X=X_test, y=y_test):
    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)

    conf_matrix = confusion_matrix(y, y_pred)
    # Plot confusion matrix
    print("\nConfusion Matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy, precision, recall, roc_auc


# Model
models = {
    "lr": LogisticRegression(solver="liblinear"),
    "dt": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(),
}

# Define the parameter grid
param_grids = {
    "lr": {"C": np.logspace(-4, 4, 20), "penalty": ["l1", "l2"]},
    "dt": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    },
    "rf": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "xgb": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.2],
    },
}

models_trained = {}
best_params = {}
evaluates = {}
for name, model in models.items():
    print(f"Training {model}...")
    # Train model
    model_trained = train_model(model, param_grids[name], verbose=0)
    # Evaluate the model
    acc, prec, rec, roc_auc = evaluate_metrics(model_trained, X_test, y_test)

    models_trained[name] = model_trained
    best_params[name] = model_trained.best_params_
    evaluates[name] = [acc, prec, rec, roc_auc]

pd.DataFrame(evaluates, index=["accuracy", "precision", "recall", "roc_auc"]).to_csv(
    "../../reports/evaluates.csv"
)

# Save models
for name, model in models_trained.items():
    joblib.dump(model.best_estimator_, f"../../models/{name}_model.pkl")

# Save best params as json
with open("best_params.json", "w") as f:
    txt = json.dumps(best_params)
    f.write(txt)
