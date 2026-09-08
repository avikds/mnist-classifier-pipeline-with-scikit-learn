"""
MNIST Classifier Pipeline with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist
import os
import tempfile
import urllib.request
import numpy as np

def load_mnist(n_train=10000, n_test=2000):
    # Cache the MNIST dataset in the system temporary directory.
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    cache_path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # Download only if the file is not already present.
    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(url, cache_path)

    # Load the cached dataset.
    with np.load(cache_path) as data:
        X_train = data["x_train"][:n_train].reshape(n_train, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)
        X_test = data["x_test"][:n_test].reshape(n_test, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 2 - binary_target
def binary_target(y, digit=5):
    # Convert y to a NumPy array, then compare each label to digit.
    return np.asarray(y) == digit

# Step 3 - train_sgd
from sklearn.linear_model import SGDClassifier

def train_sgd(X, y, random_state=42):
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf

# Step 4 - cross_val_predictions
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method
    )

# Step 5 - confusion_counts
from sklearn.metrics import confusion_matrix

def confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    return {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }

# Step 6 - precision_recall_f1
from sklearn.metrics import precision_score, recall_score, f1_score

def precision_recall_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return float(precision), float(recall), float(f1)

# Step 7 - threshold_for_precision
from sklearn.metrics import precision_recall_curve

def threshold_for_precision(y_true, scores, target=0.90):
    precisions, _, thresholds = precision_recall_curve(y_true, scores)
    idx = np.argmax(precisions[:-1] >= target)

    return float(thresholds[idx])

# Step 8 - evaluate_at_threshold
def evaluate_at_threshold(y_true, scores, threshold):
    predictions = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(predictions.sum()),
    }

# Step 9 - roc_auc
from sklearn.metrics import roc_auc_score, roc_curve

def roc_auc(y_true, scores):
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    idx = np.argmax(tpr >= 0.9)

    return {
        "auc": float(auc),
        "fpr_at_recall_90": float(fpr[idx]),
    }

# Step 10 - multiclass_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def multiclass_pipeline(random_state=42):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state)
    )

# Step 11 - multiclass_cv_accuracy
from sklearn.model_selection import cross_val_score

def multiclass_cv_accuracy(model, X, y, cv=3):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    return float(scores.mean())

# Step 12 - normalized_confusion
def normalized_confusion(y_true, y_pred):
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true"
    )

# Step 13 - most_confused_pairs
def most_confused_pairs(cm, k=3):
    cm = np.asarray(cm)
    pairs = []

    for i in range(10):
        for j in range(10):
            if i != j:
                pairs.append((i, j, float(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)

    return [
        (true_class, predicted_class, round(rate, 3))
        for true_class, predicted_class, rate in pairs[:k]
    ]

# Step 14 - multilabel_targets
def multilabel_targets(y):
    y = np.asarray(y)
    return np.column_stack([
        y >= 7,
        y % 2 == 1
    ])

# Step 15 - multilabel_knn
from sklearn.neighbors import KNeighborsClassifier

def multilabel_knn(X, Y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, Y)
    return knn


def multilabel_f1(Y_true, Y_pred):
    return float(f1_score(Y_true, Y_pred, average="macro"))

# Step 16 - final_test_accuracy
from sklearn.metrics import accuracy_score

def final_test_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred))

# Step 17 - save_and_reload_classifier
import joblib

def save_and_reload_classifier(model, path):
    joblib.dump(model, path)
    return joblib.load(path)

# Step 18 - predict_digits
def predict_digits(model, images):
    X = np.asarray(images).reshape(-1, 784).astype(np.float32)
    predictions = model.predict(X)

    return [int(prediction) for prediction in predictions]

