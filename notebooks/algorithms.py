# model_algorithms.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_models(df, y, x=None, categorical=None, test_size=0.2, random_state=42):
    """
    Train RF, SVM, and Neural Network models on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    y : str
        Target column name.
    x : list or None
        Feature columns. If None, uses all columns except y and categorical.
    categorical : str or None
        Categorical column to encode.
    test_size : float
        Test set fraction.
    random_state : int
        Random seed.
    
    Returns
    -------
    dict
        Dictionary with trained models and metrics.
    """
    # --- 1. Prepare feature columns ---
    if x is None:
        x = df.drop(columns=[y] + ([categorical] if categorical else [])).columns.tolist()
    
    X = df[x].copy()
    y_target = df[y].copy()

    # --- 2. Encode categorical if specified ---
    if categorical:
        ct = ColumnTransformer([
            ("cat", OneHotEncoder(drop='first', sparse_output=False), [categorical])
        ], remainder='passthrough')
        X = ct.fit_transform(df[[categorical] + x])
    else:
        X = X.values

    # --- 3. Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=test_size, random_state=random_state
    )

    # --- 4. Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_metrics = {
        "r2": r2_score(y_test, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "feature_importance": rf.feature_importances_
    }

    # --- 5. Support Vector Machine ---
    # Scale features for SVM
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_test_svm = scaler_svm.transform(X_test)
    
    svm = SVR(kernel='rbf')
    svm.fit(X_train_svm, y_train)
    y_pred_svm = svm.predict(X_test_svm)
    svm_metrics = {
        "r2": r2_score(y_test, y_pred_svm),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_svm))
    }

    # --- 6. Neural Network ---
    scaler_nn = StandardScaler()
    X_train_nn = scaler_nn.fit_transform(X_train)
    X_test_nn = scaler_nn.transform(X_test)
    
    nn = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=random_state)
    nn.fit(X_train_nn, y_train)
    y_pred_nn = nn.predict(X_test_nn)
    nn_metrics = {
        "r2": r2_score(y_test, y_pred_nn),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_nn))
    }

    # --- 7. Return models and metrics ---
    results = {
        "RandomForest": {"model": rf, "metrics": rf_metrics},
        "SVM": {"model": svm, "metrics": svm_metrics},
        "NeuralNetwork": {"model": nn, "metrics": nn_metrics}
    }

    return results
