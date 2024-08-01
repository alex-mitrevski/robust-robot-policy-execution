"""
Compute the metrics for the anomaly detection task using the distances and true labels.
The distances are the distances of the test frames from the nearest training frame, and
the true labels indicate whether the test frames are normal or anomalous. 
The function should generate and plot the ROC and Precision-Recall curves, 
and print the optimal thresholds for both curves. 
The function should take the distances and true labels as input
and return the optimal thresholds for both curves.
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
    plt.xlabel("Recall",fontsize=12)
    plt.ylabel("Precision",fontsize=12)
    plt.title("Precision-Recall Curve",fontsize=12)
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
        xytext=(5,-20),
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
        default=config.THRESH_CALC_DATASET_PATH
        # default="/home/bharath/AD_repo/inference/features_DINO_batch2/features_28_05/val/val_13_06_batch1_2_train"

    )
    args = parser.parse_args()
    # each json corresponds to a test file. aggregate all distances and true labels to lists
    distances = []
    true_labels = []

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

    #count 1 and 0 in true labels
    print("True labels count: ", len(true_labels))
    print("True labels count 1: ", true_labels.count(1))
    print("True labels count 0: ", true_labels.count(0))
    # import ipdb;ipdb.set_trace()
    compute_metrics(distances, true_labels)




