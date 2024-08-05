"""
Script to calculate the Precision, Recall and F1 Score for the chosen threshold

Author: Bharath Santhanam
Email:bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

References:
Sklearn library for precision,recall and F1 score calculation
1. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
2. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import argparse
import json
import os
import inference_config as config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_and_labels_path",
        type=str,
        help="path to the json file containing the distances and true labels",
        default=config.THRESH_CALC_DATASET_PATH,
    )
    parser.add_argument(
        "--threshold", type=float, help="threshold for anomaly detection", default=29.7
    )
    args = parser.parse_args()
    # each json corresponds to a test file. aggregate all distances and true labels to lists
    distances = []
    true_labels = []
    threshold = args.threshold
    # I have multiple folders under which there is json with anomaly scores and true labels
    # I need to iterate over all the json files in all the folders and get the scores and labels

    for root, dirs, files in os.walk(args.scores_and_labels_path):
        for file in files:
            if file.endswith(".json"):
                print(file)
                with open(os.path.join(root, file)) as f:
                    scores_and_labels = json.load(f)
                    distances.extend(scores_and_labels["anomaly_scores_DINO"])
                    true_labels.extend(scores_and_labels["true_labels"])

    # count 1 and 0 in true labels
    print("True labels count: ", len(true_labels))
    print("Anomalies count: ", true_labels.count(1))
    print("Nominal frames count: ", true_labels.count(0))
    # wrtie code to calculate precision, recall and F1 Score at the threshold of 26.16

    predicted_labels = [1 if dist > threshold else 0 for dist in distances]

    # directly from the sklearn library to calculate precision, recall and F1 score
    # https://scikit-learn.org/stable/api/sklearn.metrics.html
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
