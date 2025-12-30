import joblib
from sklearn.metrics import classification_report, roc_auc_score

from loader import load_data
from preprocessing import (
    split_features_target,
    scale_amount,
    stratified_split
)


def evaluate_model():
    # Load model
    model = joblib.load("../models/xgboost_fraud_model.pkl")

    # Load data
    df = load_data("../data/creditcard.csv")
    X, y = split_features_target(df)
    X_scaled, _ = scale_amount(X)
    X_train, X_test, y_train, y_test = stratified_split(X_scaled, y)

    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_probs)
    print(f"🚀 ROC-AUC Score: {auc:.4f}")


if __name__ == "__main__":
    evaluate_model()
