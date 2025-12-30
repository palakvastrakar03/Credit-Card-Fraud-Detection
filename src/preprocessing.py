import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def split_features_target(df):
    """
    Separate features and target variable
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def scale_amount(X):
    """
    Scale 'Amount' feature using RobustScaler
    """
    scaler = RobustScaler()
    X = X.copy()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    return X, scaler


def stratified_split(X, y, test_size=0.2):
    """
    Stratified train-test split to preserve class ratio
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


def apply_smote(X_train, y_train):
    """
    Apply SMOTE only on training data to handle class imbalance
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
