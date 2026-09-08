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
- [x] **13.** most_confused_pairs
- [x] **14.** multilabel_targets
- [x] **15.** multilabel_knn
- [x] **16.** final_test_accuracy
- [x] **17.** save_and_reload_classifier
- [x] **18.** predict_digits

## Results

```
MNIST slice: 10,000 train / 2,000 test images, 784 pixels each

5-detector out-of-fold: {'TN': 8872, 'FP': 265, 'FN': 161, 'TP': 702}
precision 0.726  recall 0.813  F1 0.767   (accuracy would be 0.957, 'never 5' scores 0.914)
threshold for 90% precision: 69,695 -> precision 0.900, recall 0.596, 571 flagged
ROC AUC 0.961; catching 90% of fives costs a false-positive rate of 0.101

multiclass scaled SGD: cross-validated accuracy 0.894
most confused: 7->9 6.4%, 5->8 5.1%, 3->5 4.5%
multilabel KNN (large? odd?): macro F1 0.911

TEST accuracy 0.871  (cross-validated estimate was 0.894)
served predictions on 5 raw images: [7, 2, 1, 0, 4] (truth [7, 2, 1, 0, 4])
```
