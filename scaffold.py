"""
MNIST Classifier Pipeline with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""MNIST classifier pipeline with scikit-learn (Hands-On ML, chapter 3).

Story: a 5-detector with honest out-of-fold metrics, a threshold chosen for a
target precision, ROC AUC; then ten classes with a scaled pipeline, the
confusions that remain, a multilabel KNN, and a saved model serving raw images.
"""
import os
import tempfile
import numpy as np


def main() -> None:
    data = load_mnist(n_train=10000, n_test=2000)
    X, y, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    print(f"MNIST slice: {len(X):,} train / {len(X_test):,} test images, 784 pixels each")

    # ---- 1. A 5-detector, evaluated honestly ----
    y5 = binary_target(y, digit=5)
    sgd = train_sgd(X, y5)
    pred = cross_val_predictions(sgd, X, y5)
    cells = confusion_counts(y5, pred)
    p, r, f = precision_recall_f1(y5, pred)
    print(f"\n5-detector out-of-fold: {cells}")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}   (accuracy would be {float((pred == y5).mean()):.3f}, "
          f"'never 5' scores {float((~y5).mean()):.3f})")

    # ---- 2. Choose a threshold for the product requirement ----
    scores = cross_val_predictions(sgd, X, y5, method="decision_function")
    t90 = threshold_for_precision(y5, scores, target=0.90)
    at90 = evaluate_at_threshold(y5, scores, t90)
    print(f"threshold for 90% precision: {t90:,.0f} -> precision {at90['precision']:.3f}, recall {at90['recall']:.3f}, "
          f"{at90['positives']} flagged")
    auc = roc_auc(y5, scores)
    print(f"ROC AUC {auc['auc']:.3f}; catching 90% of fives costs a false-positive rate of {auc['fpr_at_recall_90']:.3f}")

    # ---- 3. Ten classes ----
    model = multiclass_pipeline()
    cv_acc = multiclass_cv_accuracy(model, X, y)
    print(f"\nmulticlass scaled SGD: cross-validated accuracy {cv_acc:.3f}")
    pred10 = cross_val_predictions(model, X, y)
    cm = normalized_confusion(y, pred10)
    pairs = most_confused_pairs(cm, k=3)
    print("most confused:", ", ".join(f"{t}->{q} {rate:.1%}" for t, q, rate in pairs))

    # ---- 4. Two questions per image ----
    Y = multilabel_targets(y[:2000])
    knn = multilabel_knn(X[:2000], Y)
    mf1 = multilabel_f1(multilabel_targets(y_test[:500]), knn.predict(X_test[:500]))
    print(f"multilabel KNN (large? odd?): macro F1 {mf1:.3f}")

    # ---- 5. Test once, ship ----
    model.fit(X, y)
    print(f"\nTEST accuracy {final_test_accuracy(model, X_test, y_test):.3f}  (cross-validated estimate was {cv_acc:.3f})")
    path = os.path.join(tempfile.gettempdir(), "mnist_sgd_pipeline.pkl")
    served = save_and_reload_classifier(model, path)
    with np.load(os.path.join(tempfile.gettempdir(), "mnist.npz")) as z:
        imgs = z["x_test"][:5]
    print(f"served predictions on 5 raw images: {predict_digits(served, imgs)} (truth {y_test[:5].tolist()})")


if __name__ == "__main__":
    main()

