from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

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

def get_mnist():
    out_path = './sklearn_models/'
    dataset = 'mnist'
    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(f"{out_path}{dataset}"): os.mkdir(f"{out_path}{dataset}")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test, f"{out_path}{dataset}"]

def train_models():
    random_forest = RandomForestClassifier(max_depth=36, n_estimators=171, random_state=42)
    adaBoost = AdaBoostClassifier(learning_rate = 0.3845401188473625, n_estimators=100)
    support_vector_machine = SVC(probability=True)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)

    M = {
        "random_forest": random_forest, 
        "adaBoost": adaBoost, 
        "support_vector_machine": support_vector_machine, 
        "mlp": mlp, 
        "decision_tree": decision_tree 
        }

    X_train, X_test, y_train, y_test, out = get_mnist()
    ground_truths = pd.DataFrame(y_test)
    ground_truths.columns = range(ground_truths.shape[1])

    # T is a dictionary of scoring system tensors with keys {t_1, t_2, ..., t_N}
    T = classification_suite(M, X_train, y_train, X_test, y_test, out)
    
    df_reset = ground_truths.reset_index(drop=True)
    G = df_reset.iloc[:, 0].astype(int)
    return T, G

import time

start_time = time.time()
train_models()
end_time = time.time()
print(start_time - end_time)