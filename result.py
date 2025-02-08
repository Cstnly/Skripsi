import pandas as pd
import os
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


def preprocess_data(df, output_path='dataset/cleaned_data.csv'):
    # P.1.1 Drop Index Column
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
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

    # Add `SeriousDlqin2yrs` back to resampled data
    cleaned_df = pd.DataFrame(X_resampled, columns=X.columns)
    cleaned_df['SeriousDlqin2yrs'] = y_resampled

    return X_resampled, y_resampled

def main():
    df = pd.read_csv(os.path.join('dataset', 'cs-training.csv'))

    X, y = preprocess_data(df)

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

if __name__ == "__main__":
    main()