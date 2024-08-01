import copy
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
import os
from pytorch_lightning.loggers import TensorBoardLogger

import config

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

        self.criterion = DINOLoss(warmup_teacher_temp_epochs=config.WARMUP_TEACHER_TEMP_EPOCHS)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, config.MAX_EPOCHS, 0.996, 1)
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
        optim = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        return optim

    def extract_features(self, x):
        with torch.no_grad():
            features = self.student_backbone(x)
            features = features.flatten(start_dim=1)
        return features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DINO.load_from_checkpoint(checkpoint_path=config.CHECKPOINT_PATH)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(config.RESIZE),
        transforms.CenterCrop(config.CENTER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])

    features_list = []
    for img in sorted(os.listdir(config.IMAGE_PATH)):
        print(img)
        image = Image.open(os.path.join(config.IMAGE_PATH, img))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.extract_features(image_tensor)
        features_list.append(features)

    embeddings_tensor = torch.stack([embedding.squeeze().cpu() for embedding in features_list])
    print(embeddings_tensor.shape)

    torch.save(features_list, config.FEATURES_SAVE_PATH)

if __name__ == "__main__":
    main()