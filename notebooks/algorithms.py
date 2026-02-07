# algorithms.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------
# 1. Random Forest Regressor
# -------------------------
def train_random_forest(df, target, features=None, test_size=0.2, random_state=42, plot_feature_importance=True):
    if features is None:
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        features = [col for col in numeric_cols if col != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    rf = RandomForestRegressor(random_state=random_state)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    y_pred = best_rf.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    if plot_feature_importance:
        importances = best_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        plt.bar(range(len(features)), importances[indices], align="center")
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
        plt.title("Random Forest Feature Importances")
        plt.tight_layout()
        plt.show()

    print("Best RF Params:", grid_search.best_params_)
    return {"model": best_rf, "metrics": metrics}

# -------------------------
# 2. Logistic Regression
# -------------------------
def logistic_reg(df, y, x):
    """
    Logistic regression plot showing the S-curve by plotting Predicted Probability
    against the Linear Predictor (Logit), now including model results printing.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = df[x]
    y_binary = df[y]

    # Train model
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y_binary)

    # -----------------------------------------------
    #  ADDED: Print Model Results
    # -----------------------------------------------
    
    # Training accuracy
    y_pred = model.predict(X)
    accuracy = (y_pred == y_binary).mean()
    print(f"Training Accuracy: {accuracy:.3f}")

    # Model summary (Coefficients and Odds Ratios)
    summary = pd.DataFrame({
        "Feature": x,
        "Coefficient": model.coef_[0],
        "OddsRatio": np.exp(model.coef_[0])
    })
    print("\nLogistic Regression Model Summary:")
    print(summary)
    print(f"\nIntercept (b0): {model.intercept_[0]:.4f}")
    
    # -----------------------------------------------
    
    # Compute linear combination (Logit) and predicted probability
    # Logit is the input to the sigmoid function: logit = b0 + b1*x1 + ...
    logit = model.intercept_[0] + np.dot(X, model.coef_[0])
    prob_scurve = 1 / (1 + np.exp(-logit))  # P(Y=1) = Sigmoid(Logit)

    # Sort by the Logit for a smooth curve and ordered scatter points
    sorted_idx = np.argsort(logit) 

    # --- Generate data for the smooth S-curve line ---
    # Create a dense range of logit values spanning the observed range
    min_logit = logit.min()
    max_logit = logit.max()
    logit_range = np.linspace(min_logit, max_logit, 500)
    # Apply the sigmoid function to the range to get the smooth S-curve
    prob_range = 1 / (1 + np.exp(-logit_range)) 

    # --- PLOT SECTION ---
    plt.figure(figsize=(8,5))

    # 1. Plot the S-Curve (Blue Line)
    # X-axis is the Logit, Y-axis is the Predicted Probability.
    plt.plot(
        logit_range, 
        prob_range,
        color='blue', 
        lw=2, 
        label='S-curve (Predicted Probability)'
    )

    # 2. Scatter Plot: Actual Class (y-axis) vs. Logit (x-axis)
    # This shows where the binary outcomes (0 or 1) fall relative to the curve.
    plt.scatter(
        logit[sorted_idx], 
        y_binary.iloc[sorted_idx], 
        color='red', 
        alpha=0.3, 
        label='Actual class'
    )

    plt.xlabel("Linear Predictor (Logit)")
    plt.ylabel("Observed response / Predicted probability")
    plt.title("Logistic Regression S-Curve (Probability vs. Logit)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return model

# -------------------------
# 3. Neural Network Regressor
# -------------------------
def train_neural_network(df, target, features=None, test_size=0.2, random_state=42):
    if features is None:
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        features = [col for col in numeric_cols if col != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000, random_state=random_state)
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    return {"model": nn, "metrics": metrics}
