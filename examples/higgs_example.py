from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# https://www.openml.org/search?type=data&status=active&sort=date&id=45570

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
    dataset = 'Higgs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(f"{out_path}{dataset}"): os.mkdir(f"{out_path}{dataset}")
    data = fetch_openml('Higgs', version=1)
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = pd.Series(data['target'])

    # Combine X and y into a single DataFrame
    df = X.copy()
    df['target'] = y

    # Drop rows with any NaN values
    df_cleaned = df.dropna()

    # Split the DataFrame back into X and y
    y_cleaned = df_cleaned['target']
    X_cleaned = df_cleaned.drop(columns=['target'])
    X, y = X_cleaned, y_cleaned
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test, f"{out_path}{dataset}"]

def train_models():
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    adaBoost = AdaBoostClassifier(n_estimators=100)
    support_vector_machine = SVC(probability=True)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)

    M = {
        # "m1": random_forest, 
        # "m2": adaBoost, 
        # "m3": support_vector_machine, 
        "m4": mlp, 
        "m5": decision_tree 
        }

    X_train, X_test, y_train, y_test, out = get_data()
    ground_truths = pd.DataFrame(y_test)
    ground_truths.columns = range(ground_truths.shape[1])

    # T is a dictionary of scoring system tensors with keys {t_1, t_2, ..., t_N}
    T = classification_suite(M, X_train, y_train, X_test, y_test, out)

    df_reset = ground_truths.reset_index(drop=True)
    G = df_reset.iloc[:, 0].astype(int)
    G.to_csv(f'{out}/ground_truths.csv')
    return T, G

print(train_models())