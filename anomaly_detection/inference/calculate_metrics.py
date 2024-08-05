"""
Compute the metrics for the anomaly detection task using the distances and true labels.
The distances are the distances of the test frames from the nearest training frame, and
the true labels indicate whether the test frames are normal or anomalous.
The function should generate and plot the ROC and Precision-Recall curves,
and print the optimal thresholds for both curves.
Compute the precision, recall and F1 Score at the given threshold
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


def compute_metrics(distances, true_labels):
    # Generate Precision-Recall data
    precision, recall, pr_thresholds = precision_recall_curve(
        true_labels, distances, pos_label=1
    )

    pr_auc = auc(recall, precision)

    # Plot Precision-Recall Curve
    # plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="green", label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    # add a red dot for the optimal threshold at index 308 and write the threshold value

    # Finding optimal thresholds
    # # For ROC: You might choose the threshold that maximizes the TPR while minimizing the FPR
    # roc_optimal_idx = np.argmax(tpr - fpr)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    pr_optimal_idx = np.argmax(f1_scores)
    pr_optimal_threshold = pr_thresholds[pr_optimal_idx]

    print("Optimal PR Threshold:", pr_optimal_threshold)
    plt.scatter(recall[pr_optimal_idx], precision[pr_optimal_idx], color="red")

    # shrink pr_optimal_threshold to 2 decimal places
    pr_optimal_threshold = round(pr_optimal_threshold, 2)
    plt.annotate(
        f"Threshold = {pr_optimal_threshold}",
        (recall[pr_optimal_idx], precision[pr_optimal_idx]),
        textcoords="offset points",
        xytext=(260, -20),
        ha="left",
        color="red",
    )

    plt.tight_layout()
    plt.savefig("pr_curve.png")
    plt.show()


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
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
