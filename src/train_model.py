import joblib
import numpy as np

from loader import load_data
from preprocessing import (
    split_features_target,
    scale_amount,
    stratified_split
)

from xgboost import XGBClassifier


def train_xgboost_model():
    # Load data
    df = load_data("../data/creditcard.csv")
    X, y = split_features_target(df)
    X_scaled, _ = scale_amount(X)
    X_train, X_test, y_train, y_test = stratified_split(X_scaled, y)

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "../models/xgboost_fraud_model.pkl")
    print("✅ XGBoost model trained and saved successfully")


if __name__ == "__main__":
    train_xgboost_model()
