from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd
import os

def preprocess_data(X):
    # Identify categorical columns (you might need to adjust this list based on your dataset)
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
    # Apply one-hot encoding
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X

def classification_suite(M, X_train, y_train, X_test, y_test, out):
    predictions = {}
    for m in M:
        model = M[m]
        model.fit(X_train, y_train)

        class_probabilities = model.predict_proba(X_test)

        predictions[m] = pd.DataFrame(class_probabilities)
        predictions[m].to_csv(f"{out}/{m}_scores.csv")

        # Show accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{m} Accuracy: {accuracy}")

    return predictions

def get_data():
    out_path = './sklearn_models/'
    dataset = 'twitter'
    os.makedirs(f"{out_path}{dataset}", exist_ok=True)
    data = fetch_openml('Twitter-Sentiment-Analysis', version=1)
    X = data['data']
    y = data['target']

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=data['feature_names'])

    if 'airline_sentiment' in X.columns:
        y = X['airline_sentiment']
        X.drop('airline_sentiment', axis=1, inplace=True)

    # Preprocess data to handle categorical variables
    X = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, f"{out_path}{dataset}"

def train_models():
    # Define classifiers
    M = {
        "m1": RandomForestClassifier(n_estimators=100, random_state=42),
        "m2": AdaBoostClassifier(n_estimators=100),
        "m3": SVC(probability=True),
        "m4": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        "m5": DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
    }

    # Fetch the data
    X_train, X_test, y_train, y_test, out = get_data()

    # Ensure y_test is a DataFrame with the correct format
    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame(y_test)
    
    # Reset index if needed
    y_test.reset_index(drop=True, inplace=True)
    if y_test.shape[1] == 1:
        y_test.columns = ['Target']

    # Run classification
    T = classification_suite(M, X_train, y_train, X_test, y_test['Target'], out)
    return T, y_test['Target']

T, G = train_models()

print(T)