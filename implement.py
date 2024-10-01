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

# Dataset path
dataset = "/home/sonnerag/Pictures/Dataset"
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

        # Skip empty images and invalid annotations
        if img is None or len(target) == 0:
            return None, None

        annotations = {'image_id': self.ids[idx], 'annotations': target}
        encoding = self.image_processor(images=img, annotations=annotations, return_tensors="pt")

        # Ensure pixel_values are correctly formatted
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]

        return pixel_values, labels

# Check if directories exist
assert os.path.exists(TRAIN_DIRECTORY), "Train directory does not exist!"
assert os.path.exists(VAL_DIRECTORY), "Validation directory does not exist!"
assert os.path.exists(TEST_DIRECTORY), "Test directory does not exist!"

# Load pre-trained DETR model image processor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Create dataset objects
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

def collate_fn(batch):
    pixel_values = []
    valid_batch = []

    # Collect valid items (non-empty)
    for item in batch:
        if item[0] is not None:
            pixel_values.append(item[0])
            valid_batch.append(item)

    # Skip if the batch is empty (no valid images)
    if len(pixel_values) == 0:
        return None

    # Pad the pixel values
    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    labels = [item[1] for item in valid_batch]

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }

# Create DataLoader objects
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,
                              collate_fn=collate_fn,
                              batch_size=1,  # Reduced to 1 for minimal memory usage
                              shuffle=True,
                              num_workers=1,
                              drop_last=False)

VAL_DATALOADER = DataLoader(dataset=VAL_DATASET,
                            collate_fn=collate_fn,
                            batch_size=1,  # Keep batch size small for validation as well
                            num_workers=1)

# Create id2label mapping
categories = TRAIN_DATASET.coco.cats
print("Categories:")
print(categories)

id2label = {k: v['name'] for k, v in categories.items()}

print("id2label:")
print(id2label)
print(len(id2label))

# Define the model architecture and load the trained weights from the directory
device = torch.device("cpu")


MODEL_SAVE_DIR = "/home/sonnerag/Desktop/detr_model2"

# Load the saved model from the directory using Hugging Face
model = DetrForObjectDetection.from_pretrained(MODEL_SAVE_DIR)

# Move the model to the appropriate device (GPU/CPU)
model.to(device)

# Verify that the model has been loaded correctly
print(f"Model successfully loaded from {MODEL_SAVE_DIR} on {device}")

# Inference Function
def run_inference(dataloader, model, device):
    model.eval()
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)

        # Run model inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # Process outputs (this will contain boxes, logits, etc.)
        print("Output Logits:")
        print(outputs.logits)  # Print the raw logits
        print("Predicted boxes:")
        print(outputs.pred_boxes)  # Print the predicted boxes
        break  # Exit after one batch for demo purposes

# Run inference on validation data
print("Running inference on validation data...")
run_inference(VAL_DATALOADER, model, device)

# Load a specific image for analysis
image_path = "/home/sonnerag/Pictures/Dataset/test/117_jpg.rf.119dccd2483b04d8d3a8c33a1393d362.jpg"
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Set a higher confidence threshold
CONFIDENCE_THRESHOLD = 0.181  # Adjust to filter low-confidence predictions

# Perform inference with no gradient calculation (inference mode)
with torch.no_grad():
    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Post-process predictions to filter based on the confidence threshold
    target_size = torch.tensor([image.shape[:2]]).to(device)
    results = image_processor.post_process_object_detection(
        outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_size
    )[0]

# Use the Detections class to handle the model's results
detections = sv.Detections.from_transformers(transformers_results=results)

# Debugging: Check if there are any detections
print(f"Detections: {detections}")

# Create labels for detected objects
labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]

# Debugging: Check if labels are generated
print(f"Labels: {labels}")

# Create a BoxAnnotator instance to draw the boxes and labels
box_annotator = sv.BoxAnnotator()

# Annotate the image with bounding boxes and labels
image_with_detections = box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    labels=labels
)

# Display the annotated image only
cv2.imshow("Annotated Image", image_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()

