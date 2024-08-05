"""
Script to extract features from the images in the dataset and compute the anomaly scores using the nearest neighbors model

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

Description:
Takes the directory that has many folders containing datasets as input
Loads the trained DINO model
Extracts features from the images in the dataset
Normalizes the features using the mean and std of the nominal features
Computes the anomaly scores using the KNN model
Saves the features and anomaly scores in the directory specified

References:
1. Nearest neighbours estimation using Sklearn library: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
2. Anomaly detection model using LightlySSL, the source can be found at : 
https://github.com/lightly-ai/lightly/blob/master/examples/pytorch_lightning/dino.py
Lightly SSL uses open-source MIT License. The license can be found at: https://github.com/lightly-ai/lightly/blob/master/LICENSE.txt

"""

import copy
import json
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.data.dataset import LightlyDataset
import ipdb
import os
from visualize_anomalies import get_train_mean_std, normalize_features, extract_features
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Load the features
from sklearn.neighbors import NearestNeighbors
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import inference_config as config

# class direcly adapted from 
# https://github.com/lightly-ai/lightly/blob/bf3441205f73958382b83ec058f58fe3baf6a55f/examples/pytorch_lightning/dino.py#L19
class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet50 = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        backbone = nn.Sequential(*list(resnet50.children())[:-1])

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead()
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(warmup_teacher_temp_epochs=10)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 100, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim

    def extract_features(self, x):
        with torch.no_grad():
            features = self.student_backbone(x)
            # Flatten the features except for the batch dimension
            features = features.flatten(start_dim=1)
        return features


def plot_anomaly_scores(anomaly_scores, anom_frames, save_path):
    # Scatter plot of anomaly scores with red color dots
    plt.scatter(
        range(len(anomaly_scores)), anomaly_scores, c="r", label="Anomaly scores"
    )
    plt.xlabel("Image index")
    plt.ylabel("Anomaly scores")
    # Add vertical lines for the anomalous frames
    if anom_frames is not None:
        for frame in anom_frames:
            plt.axvline(x=frame, color="r", linestyle="--")
    # give the title same as the image directory
    title = save_path.split("/")[-1].split(".")[0]
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# Get the following arguments
# 1. Get directory that has many folders containing datasets
# 2. Get the directory that has the trained model
# 3. Get the file path where the nominal features are saved!
# 5. Get the location of test_frames.json
# 4. Get the directory to save the features
def main():
    parser = argparse.ArgumentParser(description="Extract features from the dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory that has many folders containing datasets",
        default=config.DATASET_DIR,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory that has the trained model",
        default=config.MODEL_DIR,
        # default="/home/bharath/AD_repo/inference/features_DINO_batch2/trained_model_DINO_7_6_2024/6_6_2024/DINO_ep100/version_0/checkpoints/epoch=99-step=2900.ckpt"
    )
    parser.add_argument(
        "--nominal_features_path",
        type=str,
        help="File path where the nominal features are saved!",
        default=config.NOMINAL_FEATURES_PATH,  # old train features with batch 1 and 2
        # default = "/home/bharath/AD_repo/inference/features_DINO_batch2/Datasets_inference/nominal_features_7_6_2024/nominal_features.pt"
    )
    parser.add_argument(
        "--anom_frames_json_path",
        type=str,
        help="Location of test_frames.json",
        default=config.ANOM_FRAMES_JSON_PATH,
    )
    parser.add_argument(
        "--save_features_dir",
        type=str,
        help="Directory to save the features",
        default=config.SAVE_FEATURES_DIR,
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    nominal_features_path = args.nominal_features_path
    save_features_dir = args.save_features_dir
    anom_frames_json_path = args.anom_frames_json_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = DINO.load_from_checkpoint(checkpoint_path=model_dir)

    model.to(device)
    model.eval()

    # Assuming you're using a CUDA device if available, else CPU
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    nom_features_tensor = extract_features(nominal_features_path)
    nominal_mean, nominal_std = get_train_mean_std(nom_features_tensor)
    nom_features_np = normalize_features(nom_features_tensor, nominal_mean, nominal_std)
    # compute the nearest neighbour
    # Initialize the NearestNeighbors model
    k = 5  # Number of neighbors to use for KNN
    knn_model = NearestNeighbors(n_neighbors=k)

    # Fit the model on the nominal features
    knn_model.fit(nom_features_np)  # nom_features or normalized_nominal_features

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        features_list = []
        anomaly_scores_list = []
        for img in sorted(os.listdir(folder_path)):
            image = Image.open(os.path.join(folder_path, img))
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.extract_features(image_tensor)

            anomalous_features_np = normalize_features(
                features.cpu(), nominal_mean, nominal_std
            )
            # Find the distance to and index of the k-th nearest neighbors in the nominal data for each test feature
            distances, indices = knn_model.kneighbors(anomalous_features_np)

            # Use the distance to the nearest neighbor as the anomaly score
            anomaly_scores = distances[:, 0]
            features_list.append(features.cpu().numpy())
            anomaly_scores_list.append(anomaly_scores[0])
        # convert the features to torch tensor, save it as .pt file in the save directory
        features_tensor = torch.tensor(features_list)
        features_save_path = os.path.join(save_features_dir, folder)
        os.makedirs(features_save_path, exist_ok=True)
        # name the .pt same as the image directory
        torch.save(features_tensor, os.path.join(features_save_path, folder + ".pt"))

        # read the json file and get the anomalous frames for the current folder
        with open(anom_frames_json_path, "r") as f:
            anom_frames_dict = json.load(f)
        anom_frames = anom_frames_dict[folder]
        # ipdb.set_trace()
        # pass the anom_frames list as the second argument to the plot_anomaly_scores function
        plot_anomaly_scores(
            anomaly_scores_list,
            anom_frames,
            os.path.join(features_save_path, folder + ".png"),
        )

        # also save anomaly scores as dictionary with the key as the "anomaly score" and the value as the list of anomaly scores, save it in the same directory as features_save_path
        anomaly_scores_dict = {"anomaly_scores_DINO": anomaly_scores_list}
        # convert the anomaly scores list to float64 and save it as json
        anomaly_scores_dict["anomaly_scores_DINO"] = [
            np.float64(i) for i in anomaly_scores_dict["anomaly_scores_DINO"]
        ]

        # get the length of the anomaly scores list
        num_frames = len(anomaly_scores_list)
        # initialize the labels list with zeros
        true_labels = [0] * num_frames
        # update the labels list with 1 for the anomalous frames

        for i in range(0, len(anom_frames), 2):
            start_index = anom_frames[i]
            end_index = anom_frames[i + 1]
            for j in range(start_index, end_index + 1):
                true_labels[j] = 1

        # save the true labels in the json
        anomaly_scores_dict["true_labels"] = true_labels

        with open(
            os.path.join(features_save_path, folder + "_anomaly_scores.json"), "w"
        ) as f:
            json.dump(anomaly_scores_dict, f)


if __name__ == "__main__":
    main()
