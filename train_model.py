import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def main():
    # 1. Data loading
    data = pd.read_csv('heart.csv')

    # 2. Show top 5 records
    print("--- Top 5 Records ---")
    print(data.head())

    # 3. Identify features and target
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    target_col = 'target'
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    print("\nNumerical Features:", num_cols)
    print("Target Variable:", target_col)

    # 4. Handle missing values
    print("\nMissing Values Count:")
    print(data.isna().sum())
    data.dropna(inplace=True)

    # 5. Train-test split (80-20)
    features = data.drop(columns=[target_col])
    labels = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=7)

    # Model training: Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluation
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\nAccuracy Score: {acc * 100:.2f}%")

    # Serialization using Joblib
    joblib.dump(clf, 'model.pkl')
    print("Model serialized to model.pkl successfully.")

if __name__ == "__main__":
    main()
