# MNIST Classifier Pipeline with Scikit-Learn

Chapter 3 of Hands-On Machine Learning as a practitioner runs it: load MNIST, train an SGD classifier for one digit, get honest out-of-fold predictions with cross_val_predict, read the confusion matrix, precision, recall and F1, pick a decision threshold for a target precision from the precision-recall curve, measure ROC AUC, then go multiclass with a scaled pipeline, analyze the normalized confusion matrix for the most confused pairs, train a multilabel KNN, and finally save, reload and serve the classifier on raw 28x28 images.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_mnist
- [x] **2.** binary_target
- [x] **3.** train_sgd
- [x] **4.** cross_val_predictions
- [x] **5.** confusion_counts
- [x] **6.** precision_recall_f1
- [x] **7.** threshold_for_precision
- [x] **8.** evaluate_at_threshold
- [x] **9.** roc_auc
- [x] **10.** multiclass_pipeline
- [x] **11.** multiclass_cv_accuracy
- [x] **12.** normalized_confusion
- [ ] **13.** most_confused_pairs
- [ ] **14.** multilabel_targets
- [ ] **15.** multilabel_knn
- [ ] **16.** final_test_accuracy
- [ ] **17.** save_and_reload_classifier
- [ ] **18.** predict_digits

---

Built on Deep-ML.
