"""
Script to calculate the optimal threshold for anomaly detection using Precision-Recall curve plotted for various thresholds.
Optimal trheshold is the one that maximizes the F1 score.

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

References:
Sklearn library for precision recall curve and AUC calculation
1. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
2. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
"""

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
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
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=12)
    plt.legend(loc="lower left")

    f1_scores = 2 * (precision * recall) / (precision + recall)
    pr_optimal_idx = np.argmax(f1_scores)
    pr_optimal_threshold = pr_thresholds[pr_optimal_idx]

    # calculate area under PR curve
    pr_auc = auc(recall, precision)
    print("PR AUC:", pr_auc)
    print("Optimal PR Threshold:", pr_optimal_threshold)
    plt.scatter(recall[pr_optimal_idx], precision[pr_optimal_idx], color="red")

    # shrink pr_optimal_threshold to 2 decimal places
    pr_optimal_threshold = round(pr_optimal_threshold, 2)
    plt.annotate(
        f"Threshold: \n{pr_optimal_threshold}",
        # f"Threshold: \n{29.70}",
        (recall[pr_optimal_idx], precision[pr_optimal_idx]),
        textcoords="offset points",
        xytext=(5, -20),
        ha="left",
        color="red",
    )

    plt.tight_layout()
    plt.savefig("pr_curve_8_7_2024.pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_and_labels_path",
        type=str,
        help="path to the json file containing the distances and true labels",
        default=config.THRESH_CALC_DATASET_PATH,
    )
    args = parser.parse_args()
    # each json corresponds to a test file. aggregate all distances and true labels to lists
    distances = []
    true_labels = []

    # Have multiple folders under which there is json with anomaly scores and true labels
    # need to iterate over all the json files in all the folders and get the scores and labels

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
    print("True labels count 1: ", true_labels.count(1))
    print("True labels count 0: ", true_labels.count(0))
    # import ipdb;ipdb.set_trace()
    compute_metrics(distances, true_labels)
