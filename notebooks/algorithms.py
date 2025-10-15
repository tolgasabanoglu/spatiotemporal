# algorithms.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def train_models(df, y, x=None, test_size=0.2, random_state=42, logistic_threshold=45, plot_feature_importance=True):
    """
    Train RF, SVM, Neural Network, and Logistic Regression models on a dataset.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    y : str
        Target column name.
    x : list or None
        Feature columns. If None, uses all numeric columns except y.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.
    logistic_threshold : int
        Threshold for binary target in Logistic Regression.
    plot_feature_importance : bool
        If True, plot RF feature importances.
    Returns
    -------
    dict : dictionary containing models and metrics
    """
    # --- 1. Prepare feature columns ---
    if x is None:
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        x = [col for col in numeric_cols if col != y]

    X = df[x].copy()
    y_target = df[y].copy()

    # --- 2. Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=test_size, random_state=random_state, shuffle=False
    )

    results = {}

    # --- 3. Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_metrics = {
        "r2": r2_score(y_test, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "mae": mean_absolute_error(y_test, y_pred_rf)
    }
    results["RandomForest"] = {"model": rf, "metrics": rf_metrics}

    if plot_feature_importance:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        plt.title("Random Forest Feature Importances")
        plt.bar(range(len(x)), importances[indices], align="center")
        plt.xticks(range(len(x)), [x[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    # --- 4. Support Vector Machine ---
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_test_svm = scaler_svm.transform(X_test)
    svm = SVR(kernel='rbf')
    svm.fit(X_train_svm, y_train)
    y_pred_svm = svm.predict(X_test_svm)
    results["SVM"] = {
        "model": svm,
        "metrics": {
            "r2": r2_score(y_test, y_pred_svm),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_svm)),
            "mae": mean_absolute_error(y_test, y_pred_svm)
        }
    }

    # --- 5. Neural Network ---
    scaler_nn = StandardScaler()
    X_train_nn = scaler_nn.fit_transform(X_train)
    X_test_nn = scaler_nn.transform(X_test)
    nn = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000, random_state=random_state)
    nn.fit(X_train_nn, y_train)
    y_pred_nn = nn.predict(X_test_nn)
    results["NeuralNetwork"] = {
        "model": nn,
        "metrics": {
            "r2": r2_score(y_test, y_pred_nn),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_nn)),
            "mae": mean_absolute_error(y_test, y_pred_nn)
        }
    }

    # --- 6. Logistic Regression ---
    # Convert target to binary
    y_binary = (y_target > logistic_threshold).astype(int)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, shuffle=False
    )
    logreg = LogisticRegression(max_iter=2000, random_state=random_state)
    logreg.fit(X_train_lr, y_train_lr)
    y_pred_lr = logreg.predict(X_test_lr)
    results["LogisticRegression"] = {
        "model": logreg,
        "metrics": {
            "accuracy": (y_pred_lr == y_test_lr).mean()
        }
    }

    return results
