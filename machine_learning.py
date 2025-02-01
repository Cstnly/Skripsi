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

def check_missing_value(df):
    missing_values = df.isnull().sum()
    plt.figure(figsize=(10, 6))
    missing_values.plot(kind='barh', color='skyblue')
    plt.title("Missing Values per Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.yticks(fontsize=8)
    plt.subplots_adjust(left=0.3)

    path = os.path.join('preprocessing_figure', "missing_values_per_column.jpg")
    plt.savefig(path, format='jpg')
    plt.close()
    return

def draw_corr_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')

    path = os.path.join('preprocessing_figure', "correlation_matrix.jpg")

    plt.savefig(path)
    plt.close()
    return

def draw_box_plot(df, cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box plot of {col}')
        
        path = os.path.join('preprocessing_figure', f"box_plot_{col}.jpg")
        
        plt.savefig(path)
        plt.close()
    return

def distribution_class(y, y_resampled, save_path='preprocessing_figure'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Figure 1: Distribusi kelas sebelum resample
    plt.figure(figsize=(7, 5))
    sns.countplot(x=y, hue=y, palette='viridis', legend=False)
    plt.title("Distribusi Kelas Sebelum Resample")
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Data')
    plt.text(0, y.value_counts()[0] + 50, f"Total: {y.value_counts()[0]}", ha='center', va='bottom', fontsize=12)
    plt.text(1, y.value_counts()[1] + 50, f"Total: {y.value_counts()[1]}", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    save_file1 = os.path.join(save_path, 'distribusi_kelas_sebelum_resample.png')
    plt.savefig(save_file1)
    
    # Figure 2: Distribusi kelas setelah resample
    plt.figure(figsize=(7, 5))
    sns.countplot(x=y_resampled, hue=y_resampled, palette='viridis', legend=False)
    plt.title("Distribusi Kelas Setelah Resample")
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Data')
    plt.text(0, y_resampled.value_counts()[0] + 50, f"Total: {y_resampled.value_counts()[0]}", ha='center', va='bottom', fontsize=12)
    plt.text(1, y_resampled.value_counts()[1] + 50, f"Total: {y_resampled.value_counts()[1]}", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    save_file2 = os.path.join(save_path, 'distribusi_kelas_setelah_resample.png')
    plt.savefig(save_file2)

# def remove_outliers_iqr(df, col):
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("ROC AUC Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

def preprocess_data(df, output_path='dataset/cleaned_data.csv'):
    # P.1.1 Drop Index Column
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Check missing value
    check_missing_value(df)
    
    # P.1.2 Filling missing value with median

    # MonthlyIncome
    # High Std & Very High Max compared to the mean and median -> right Skewed
    # Since the data is more likely right Skewed (many outliers), using median is more robust than mean
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

    # NumberOfDependents
    # Heavily left skewed towards 0, this suggests that most individuals have no dependents.
    # Using mean will not be suitable, as it would fill them with 0.757, which is not a typical value in the distribution. this will introduce bias/distortion
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])

    # Removing outliers in RevolvingUtilizationOfUnsecuredLines
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 10]

    # Split the independent(X) and dependent(y)
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']

    # Apply SMOTE to balance the SeriousDlqin2yrs column
    smote = SMOTE(random_state=32)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Plot distribution class before and after resample
    distribution_class(y, y_resampled)

    # Add `SeriousDlqin2yrs` back to resampled data and save to CSV
    cleaned_df = pd.DataFrame(X_resampled, columns=X.columns)
    cleaned_df['SeriousDlqin2yrs'] = y_resampled

    # Save cleaned data to CSV
    cleaned_df.to_csv(output_path, index=False)

    return X_resampled, y_resampled

def train_model(X_train, y_train):
    # P.2.1 Random Forest Model Initialization & P.2.2 Set Default Parameter for RF Model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=32,
        n_jobs=-1
    )
    # P.2.3 Train Model
    rf.fit(X_train, y_train)
    return rf

def save_model_before_tune(model, model_path='model_before_tune/random_forest_model.joblib'):
    joblib.dump(model, model_path)

def save_model_after_tune(model, best_params, model_path='model_after_tune/best_random_forest_model.joblib', params_path = 'model_after_tune/best_params.json'):
    joblib.dump(model, model_path)

    with open(params_path, 'w') as f:
        json.dump(best_params, f)

def load_model(model_path='random_forest_model.joblib'):
    model = joblib.load(model_path)
    return model

def main():
    os.makedirs('preprocessing_figure', exist_ok=True)
    os.makedirs('model_after_tune', exist_ok=True)
    os.makedirs('model_before_tune', exist_ok=True)
    os.makedirs('cv_result', exist_ok=True)

    df = read_dataset(os.path.join('dataset', 'cs-training.csv'))
    
    # P.1.1 - P.1.6
    X, y = preprocess_data(df)

    # P.1.7 Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    # # P.2.1 - P.2.3
    start_time_train = time.time()
    model = train_model(X_train, y_train)
    train_duration = time.time() - start_time_train
    print(f"Training model selesai dalam {train_duration:.2f} detik.")
    # save_model_before_tune(model)
    
    # # Prediksi pada data uji tanpa tuning
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # # P.2.4 Evaluate RF Model Without tuning
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_pred_proba))

    # P.2.5 Set Hyperparameter Grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 5, 10, 20],
        'max_features': [5, 'sqrt', 'log2'],
    }
    
    # P.2.6 Set Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=3, n_jobs=-1)

    # P.2.7 Grid Search Tuning
    start_time_grid_search = time.time()
    grid_search.fit(X_train, y_train)
    grid_search_duration = time.time() - start_time_grid_search
    print(f"Grid search selesai dalam {grid_search_duration:.2f} detik.")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv('cv_result/grid_search_results.csv', index=False)

    print("Grid search results saved to 'grid_search_results.csv'.")

    # P.2.8 Best Parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # P.2.9 Evaluate Tuned RF Model
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best))
    print("\nAccuracy Score (Best Model):")
    print(accuracy_score(y_test, y_pred_best))
    print("\nROC AUC Score (Best Model):")
    print(roc_auc_score(y_test, y_pred_proba_best))

    save_model_after_tune(best_model, best_params)

    plot_learning_curve(model, "Learning Curve (Random Forest, Before Tuning)", X_train, y_train, cv=cv, n_jobs=-1)
    plt.savefig('model_before_tune/learning_curve_rf_before_tuning.jpg')
    plot_learning_curve(best_model, "Learning Curve (Random Forest, After Tuning)", X_train, y_train, cv=cv, n_jobs=-1)
    plt.savefig('model_after_tune/learning_curve_rf_after_tuning.jpg')

if __name__ == '__main__':
    main()