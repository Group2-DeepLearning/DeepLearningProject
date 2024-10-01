import torch
import supervision as sv
import transformers
import pytorch_lightning
import os
import torchvision
import cv2
import random
import numpy as np

from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor

# Link for the dataset: https://universe.roboflow.com/roboflow-100/bone-fracture-7fy1g
# Download the dataset with COCO format

dataset = "/home/sonnerag/Pictures/Dataset"  # dataset folder

ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "valid")
TEST_DIRECTORY = os.path.join(dataset, "test")

# Class for loading the dataset
class CustomCocoDetection(CocoDetection):
    def __init__(self, root: str, annFile: str, image_processor):
        super().__init__(root, annFile)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        annotations = {'image_id': self.ids[idx], 'annotations': target}
        encoding = self.image_processor(images=img, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]
        return pixel_values, labels

# Check if directories exist
assert os.path.exists(TRAIN_DIRECTORY), "Train directory does not exist!"
assert os.path.exists(VAL_DIRECTORY), "Validation directory does not exist!"
assert os.path.exists(TEST_DIRECTORY), "Test directory does not exist!"

# Load pre-trained DETR model image processor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Create the dataset objects
TRAIN_DATASET = CustomCocoDetection(root=TRAIN_DIRECTORY,
                                    annFile=os.path.join(TRAIN_DIRECTORY, ANNOTATION_FILE_NAME),
                                    image_processor=image_processor)
VAL_DATASET = CustomCocoDetection(root=VAL_DIRECTORY,
                                  annFile=os.path.join(VAL_DIRECTORY, ANNOTATION_FILE_NAME),
                                  image_processor=image_processor)
TEST_DATASET = CustomCocoDetection(root=TEST_DIRECTORY,
                                   annFile=os.path.join(TEST_DIRECTORY, ANNOTATION_FILE_NAME),
                                   image_processor=image_processor)

# Print the number of images in each dataset
print("Number of training images: ", len(TRAIN_DATASET))
print("Number of validation images: ", len(VAL_DATASET))
print("Number of test images: ", len(TEST_DATASET))

# Visualize the images
# Select a random image ID from the dataset
image_id = random.choice(TRAIN_DATASET.ids)
print("Image id: {}".format(image_id))

# Load the image with annotation
image_info = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image_info['file_name'])
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise ValueError(f"Image at path {image_path} could not be loaded!")

# Extract the bounding boxes (xyxy) and class IDs from the COCO annotations
xyxy = []
class_ids = []

# COCO annotations use 'bbox' in the format (x_min, y_min, width, height)
# Convert it to (x_min, y_min, x_max, y_max)
for ann in annotations:
    x_min, y_min, width, height = ann['bbox']
    x_max = x_min + width
    y_max = y_min + height
    xyxy.append([x_min, y_min, x_max, y_max])
    class_ids.append(ann['category_id'])  # Get the class ID

# Convert to numpy arrays as required by sv.Detections
xyxy = np.array(xyxy)
class_ids = np.array(class_ids)

# Create the detections object
detections = sv.Detections(xyxy=xyxy, class_id=class_ids)

# Get category names
categories = TRAIN_DATASET.coco.cats
print("Categories: ", categories)

# Map category IDs to labels
id2label = {k: v['name'] for k, v in categories.items()}

# Collect labels from detections
labels = [id2label[class_id] for class_id in class_ids]

# Print the id2label mapping and detected labels
print(id2label)
print(labels)

# Annotate the image using Supervision
box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

# Display the annotated image
cv2.imshow("Annotated Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
