from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from features import FEATURES
import pandas as pd


def train_classification_model(df, verbose=True):
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=1,
        eval_metric="logloss"
    )

    model.fit(X, y)

    if verbose:
        preds = model.predict(X)
        acc = accuracy_score(y, preds)

        print("\n--- CLASSIFICATION MODEL ---")
        print(f"Train Accuracy: {acc:.3f}")

        importance = model.feature_importances_
        feat_imp = pd.Series(importance, index=FEATURES).sort_values(ascending=False)

        print("\n--- FEATURE IMPORTANCE ---")
        print(feat_imp)

    return model


def train_regression_model(df, verbose=True):
    df = df.dropna()

    X = df[FEATURES]
    y = df["return"]

    model = XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=1
    )

    model.fit(X, y)

    if verbose:
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        print("\n--- REGRESSION MODEL ---")
        print(f"MSE: {mse:.6f}")

    return model