from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def classification_suite(M, X_train, y_train, X_test, y_test, out):
    predictions = {}
    for m in M:
        model = M[m]
        if m == "cnn":
            # Convert DataFrame to numpy array and reshape for CNN
            X_train_cnn = X_train.values.reshape(-1, 28, 28, 1)
            X_test_cnn = X_test.values.reshape(-1, 28, 28, 1)
            y_train_cnn = to_categorical(y_train)
            y_test_cnn = to_categorical(y_test)

            model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=128, verbose=1, validation_data=(X_test_cnn, y_test_cnn))
            class_probabilities = model.predict(X_test_cnn)
        else:
            model.fit(X_train, y_train)
            class_probabilities = model.predict_proba(X_test)

        predictions[m] = pd.DataFrame(class_probabilities)
        predictions[m].to_csv(f"{out}/{m}_scores.csv")

        # Show accuracy
        if m == "cnn":
            y_pred = np.argmax(class_probabilities, axis=1)
        else:
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
    X = X / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype(int)  # Use built-in int instead of np.int
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test, f"{out_path}{dataset}"]


def build_cnn():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models():
    random_forest = RandomForestClassifier(max_depth=36, n_estimators=171, random_state=42)
    adaBoost = AdaBoostClassifier(learning_rate=0.3845401188473625, n_estimators=100)
    support_vector_machine = SVC(probability=True)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
    cnn = build_cnn()

    M = {
        # "random_forest": random_forest, 
        # "adaBoost": adaBoost, 
        # "support_vector_machine": support_vector_machine, 
        # "mlp": mlp, 
        # "decision_tree": decision_tree,
        "cnn": cnn
    }

    X_train, X_test, y_train, y_test, out = get_mnist()
    ground_truths = pd.DataFrame(y_test)
    ground_truths.columns = range(ground_truths.shape[1])

    T = classification_suite(M, X_train, y_train, X_test, y_test, out)
    
    df_reset = ground_truths.reset_index(drop=True)
    G = df_reset.iloc[:, 0].astype(int)
    return T, G

import time

start_time = time.time()
train_models()
end_time = time.time()
print(f"Total time: {end_time - start_time} seconds")
