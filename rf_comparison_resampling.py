import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import json

def read_dataset(path):
    return pd.read_csv(path)

def preprocess_data(df, apply_smote=False):
    # Drop Index Column
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Fill Missing Values
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])
    
    # Remove Outliers
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 10]
    
    # Split Features and Target
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    
    # Apply SMOTE if needed
    if apply_smote:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    return X, y

def evaluate_model(X, y, random_state=42):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=52,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability scores for ROC AUC
    
    # Metrics
    report = classification_report(y_test, y_pred, target_names=["Tidak Gagal Bayar", "Gagal Bayar"], output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Tidak Gagal Bayar", "Gagal Bayar"]))
    print(f"ROC AUC Score: {auc:.4f}")
    
    return report, auc

def plot_comparison(metrics_before, metrics_after, metric_names, class_name):
    x = range(len(metric_names))
    
    # Plotting data
    plt.figure(figsize=(10, 6))
    plt.bar(x, metrics_before, width=0.4, label='Tanpa Resampling', align='center')
    plt.bar([p + 0.4 for p in x], metrics_after, width=0.4, label='Dengan Resampling', align='center')
    
    plt.xticks([p + 0.2 for p in x], metric_names)
    plt.xlabel("Metrik")
    plt.ylabel("Skor")
    plt.title(f"Perbandingan Performa Model untuk {class_name}")
    plt.legend()
    plt.show()

def main():
    # Load dataset
    filepath = os.path.join('dataset', 'cs-training.csv')
    df = read_dataset(filepath)
    
    # Preprocess without resampling
    print("### Evaluation Without Resampling ###")
    X_original, y_original = preprocess_data(df, apply_smote=False)
    report_original, auc_original = evaluate_model(X_original, y_original)
    
    # Preprocess with resampling (SMOTE)
    print("\n### Evaluation With Resampling ###")
    X_resampled, y_resampled = preprocess_data(df, apply_smote=True)
    report_resampled, auc_resampled = evaluate_model(X_resampled, y_resampled)
    
    metrics_before_class_0 = {
        'precision': report_original['Tidak Gagal Bayar']['precision'],
        'recall': report_original['Tidak Gagal Bayar']['recall'],
        'f1-score': report_original['Tidak Gagal Bayar']['f1-score'],
        'auc': auc_original
    }

    metrics_after_class_0 = {
        'precision': report_resampled['Tidak Gagal Bayar']['precision'],
        'recall': report_resampled['Tidak Gagal Bayar']['recall'],
        'f1-score': report_resampled['Tidak Gagal Bayar']['f1-score'],
        'auc': auc_resampled
    }

    metrics_before_class_1 = {
        'precision': report_original['Gagal Bayar']['precision'],
        'recall': report_original['Gagal Bayar']['recall'],
        'f1-score': report_original['Gagal Bayar']['f1-score'],
        'auc': auc_original
    }

    metrics_after_class_1 = {
        'precision': report_resampled['Gagal Bayar']['precision'],
        'recall': report_resampled['Gagal Bayar']['recall'],
        'f1-score': report_resampled['Gagal Bayar']['f1-score'],
        'auc': auc_resampled
    }

    metric_names = ["Precision", "Recall", "F1-Score", "ROC AUC"]

    metrics_before_values_class_0 = [
        metrics_before_class_0['precision'],
        metrics_before_class_0['recall'],
        metrics_before_class_0['f1-score'],
        metrics_before_class_0['auc']
    ]

    metrics_after_values_class_0 = [
        metrics_after_class_0['precision'],
        metrics_after_class_0['recall'],
        metrics_after_class_0['f1-score'],
        metrics_after_class_0['auc']
    ]

    metrics_before_values_class_1 = [
        metrics_before_class_1['precision'],
        metrics_before_class_1['recall'],
        metrics_before_class_1['f1-score'],
        metrics_before_class_1['auc']
    ]

    metrics_after_values_class_1 = [
        metrics_after_class_1['precision'],
        metrics_after_class_1['recall'],
        metrics_after_class_1['f1-score'],
        metrics_after_class_1['auc']
    ]

    plot_comparison(metrics_before_values_class_0, metrics_after_values_class_0, metric_names, "Tidak Gagal Bayar")

    plot_comparison(metrics_before_values_class_1, metrics_after_values_class_1, metric_names, "Gagal Bayar")

if __name__ == '__main__':
    main()


