import pandas as pd
import os
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

def main():
    # df = pd.read_csv(os.path.join('dataset', 'cs-training.csv'))

    data_clean = pd.read_csv(os.path.join('dataset', 'cs-training.csv'))
    data_clean = pd.DataFrame(data_clean)

    X = data_clean.drop('SeriousDlqin2yrs', axis=1)
    y = data_clean['SeriousDlqin2yrs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    # 1. Model Before Tune
    model_before_gcv = joblib.load('model_before_tune/random_forest_model.joblib')

    y_pred = model_before_gcv.predict(X_test)
    y_pred_proba = model_before_gcv.predict_proba(X_test)[:, 1]

    # evaluate
    print('Model Before Tune')
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_pred_proba))

    # 2. Model After Tune
    model_after_gcv = joblib.load('model_after_tune/best_random_forest_model.joblib')

    y_pred = model_after_gcv.predict(X_test)
    y_pred_proba = model_after_gcv.predict_proba(X_test)[:, 1]

    # evaluate
    print('Model After Tune')
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_pred_proba))

