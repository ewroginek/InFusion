from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os

def classification_suite(models, X_train, y_train, X_test, out):
    predictions = {}
    for M in models:
        model = models[M]
        model.fit(X_train, y_train)

        class_probabilities = model.predict_proba(X_test)
        pd.DataFrame(class_probabilities).to_csv(f"{out}/{M}_scores.csv")

        predictions[M] = class_probabilities

        # Show accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{M} Accuracy: {accuracy}")

    return predictions

def get_lidar_data(num):
    # Obtained from this dataset: https://www.kaggle.com/datasets/mexwell/lidar-data-for-tree-species-classification?resource=download
    out_path = './sklearn_models/'
    dataset = f'lidar_trees_classification_d{num}'
    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(f"{out_path}{dataset}"): os.mkdir(f"{out_path}{dataset}") 

    d = pd.read_csv(f'./archive/dataset_{num}_training_test.csv', sep=";")

    # Automatically map labels in "SP3" to numbers
    labels, uniques = pd.factorize(d['SP3'])
    d['SP3'] = labels

    # Split the dataset based on the "Set" column
    train_d = d[d['Set'] == 'Training']
    test_d = d[d['Set'] == 'Test']

    X_train = train_d.iloc[:, :-2]
    X_test = test_d.iloc[:, :-2]
    y_train = train_d['SP3']
    y_test = test_d['SP3']
    return [X_train, X_test, y_train, y_test, f"{out_path}{dataset}"]

def get_mnist():
    out_path = './sklearn_models/'
    dataset = 'mnist'
    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(f"{out_path}{dataset}"): os.mkdir(f"{out_path}{dataset}")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test, f"{out_path}{dataset}"]

def get_shuttle_logs():
    out_path = './sklearn_models/'
    dataset = 'shuttle_logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(f"{out_path}{dataset}"): os.mkdir(f"{out_path}{dataset}") 
    # Starlog dataset  
    statlog_shuttle = fetch_ucirepo(id=148) 
    # # data (as pandas dataframes) 
    X = statlog_shuttle.data.features 
    y = statlog_shuttle.data.targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test, f"{out_path}{dataset}"]

# Also consider stellar classification: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data
# Also consider emotion classification: https://www.kaggle.com/code/bhavikjikadara/emotions-classification-lstm-nlp-94

# feature_data = [get_lidar_data(1), get_lidar_data(2), get_mnist(), get_shuttle_logs()]
feature_data= [get_shuttle_logs()]

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
adaBoost = AdaBoostClassifier(n_estimators=100)
support_vector_machine = SVC(probability=True)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)

models = {
    "RandomForest": random_forest, 
    "AdaBoost": adaBoost, 
    "SVM": support_vector_machine, 
    "MLP": mlp, 
    "DecisionTree": decision_tree 
    }

for data in feature_data:
    X_train, X_test, y_train, y_test, out = data
    preds = classification_suite(models, X_train, y_train, X_test, out)

    ground_truths = pd.DataFrame(y_test)
    ground_truths.columns = range(ground_truths.shape[1])

    # Different labels needed for shuttle logs
    if out == './sklearn_models/shuttle_logs':
        ground_truths = ground_truths.reset_index(drop=True) # for shuttle_logs
        ground_truths = ground_truths[0] - 1 # for shuttle_logs
    
    ground_truths.to_csv(f'{out}/ground_truths.csv')

