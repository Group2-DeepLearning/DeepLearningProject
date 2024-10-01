import torch
import supervision as sv
import transformers
import pytorch_lightning as pl
import os
import torchvision
import cv2
import random
import numpy as np
from torch.utils.checkpoint import checkpoint
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer


dataset = "/home/sonnerag/Pictures/Dataset"

ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "valid")
TEST_DIRECTORY = os.path.join(dataset, "test")

class CustomCocoDetection(CocoDetection):
    def __init__(self, root: str, annFile: str, image_processor):
        super().__init__(root, annFile)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if img is None or len(target) == 0:
            return None, None

        annotations = {'image_id': self.ids[idx], 'annotations': target}
        encoding = self.image_processor(images=img, annotations=annotations, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]

        return pixel_values, labels

assert os.path.exists(TRAIN_DIRECTORY), "Train directory does not exist!"
assert os.path.exists(VAL_DIRECTORY), "Validation directory does not exist!"
assert os.path.exists(TEST_DIRECTORY), "Test directory does not exist!"

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

TRAIN_DATASET = CustomCocoDetection(root=TRAIN_DIRECTORY,
                                    annFile=os.path.join(TRAIN_DIRECTORY, ANNOTATION_FILE_NAME),
                                    image_processor=image_processor)
VAL_DATASET = CustomCocoDetection(root=VAL_DIRECTORY,
                                  annFile=os.path.join(VAL_DIRECTORY, ANNOTATION_FILE_NAME),
                                  image_processor=image_processor)
TEST_DATASET = CustomCocoDetection(root=TEST_DIRECTORY,
                                   annFile=os.path.join(TEST_DIRECTORY, ANNOTATION_FILE_NAME),
                                   image_processor=image_processor)

print("Number of training images: ", len(TRAIN_DATASET))
print("Number of validation images: ", len(VAL_DATASET))
print("Number of test images: ", len(TEST_DATASET))

def collate_fn(batch):
    pixel_values = []
    valid_batch = []

    for item in batch:
        if item[0] is not None:
            pixel_values.append(item[0])
            valid_batch.append(item)

    if len(pixel_values) == 0:
        return None

    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in valid_batch]

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,
                              collate_fn=collate_fn,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              drop_last=False)

VAL_DATALOADER = DataLoader(dataset=VAL_DATASET,
                            collate_fn=collate_fn,
                            batch_size=1,
                            num_workers=1)

categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}
label2id = {v: k for k, v in id2label.items()}

class Detr(pl.LightningModule):
    def __init__(self, id2label, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=len(id2label),
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        if batch is None:
            return None

        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        if loss is not None:
            self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        if loss is not None:
            self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad], "lr": self.lr},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone}
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

model = Detr(id2label=id2label, lr=5e-6, lr_backbone=5e-7, weight_decay=1e-4)  # Lower learning rates

log_dir = '/home/sonnerag/Pictures/Dataset/lightning_logs'
os.makedirs(log_dir, exist_ok=True)
csv_logger = CSVLogger(log_dir, name='training_logs')

trainer = Trainer(
    devices=1,
    accelerator="cpu",
    max_epochs=2,
    gradient_clip_val=0.1,
    accumulate_grad_batches=1,
    log_every_n_steps=10,
    logger=csv_logger,
    default_root_dir=log_dir
)

trainer.fit(model)

MODEL_SAVE_DIR = "/home/sonnerag/Desktop/detr_model2"
model.model.save_pretrained(MODEL_SAVE_DIR)
print(f"Model saved to {MODEL_SAVE_DIR}")
