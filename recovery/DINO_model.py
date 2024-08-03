"""
DINO Class to extract image features during deployment on the real robot.
Uses pretrained checkpoint to extract features from the images.

Dependencies:
- Pytorch Lightning
- Pytorch
- lightly
- torchvision

Authors: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

References:
This script is adapted from the official lightly documentation:
https://docs.lightly.ai/self-supervised-learning/examples/dino.html

"""

import copy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
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
