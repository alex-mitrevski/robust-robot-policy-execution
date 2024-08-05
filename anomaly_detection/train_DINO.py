"""
Script to train the DINO model on the nominal images
Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

References:
This class DINO is directly adapted from the Lightly SSL implementation of DINO, given in the 
official lightly SSL reposiory. The source can be found at : 
https://github.com/lightly-ai/lightly/blob/master/examples/pytorch_lightning/dino.py
Lightly SSL uses open-source MIT License. The license can be found at: https://github.com/lightly-ai/lightly/blob/master/LICENSE.txt
"""

import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.data.dataset import LightlyDataset
from pytorch_lightning.loggers import TensorBoardLogger

import config

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

        self.criterion = DINOLoss(
            warmup_teacher_temp_epochs=config.WARMUP_TEACHER_TEMP_EPOCHS
        )

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
        return torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)


def main():
    model = DINO()

    transform = DINOTransform()
    dataset = LightlyDataset(config.DATASET_PATH, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger = TensorBoardLogger(save_dir=config.LOGS_DIR, name=config.EXPERIMENT_NAME)

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        devices=config.DEVICES,
        accelerator=accelerator,
        strategy=config.STRATEGY,
        sync_batchnorm=config.SYNC_BATCHNORM,
        replace_sampler_ddp=config.REPLACE_SAMPLER_DDP,
        logger=logger,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
    )

    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
