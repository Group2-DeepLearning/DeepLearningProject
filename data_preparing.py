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



dataset = "/home/sonnerag/Pictures/Dataset"  # dataset folder

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


image_id = random.choice(TRAIN_DATASET.ids)
print("Image id: {}".format(image_id))


image_info = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image_info['file_name'])
image = cv2.imread(image_path)


if image is None:
    raise ValueError(f"Image at path {image_path} could not be loaded!")


xyxy = []
class_ids = []


for ann in annotations:
    x_min, y_min, width, height = ann['bbox']
    x_max = x_min + width
    y_max = y_min + height
    xyxy.append([x_min, y_min, x_max, y_max])
    class_ids.append(ann['category_id'])


xyxy = np.array(xyxy)
class_ids = np.array(class_ids)


detections = sv.Detections(xyxy=xyxy, class_id=class_ids)


categories = TRAIN_DATASET.coco.cats
print("Categories: ", categories)


id2label = {k: v['name'] for k, v in categories.items()}


labels = [id2label[class_id] for class_id in class_ids]


print(id2label)
print(labels)


box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)



cv2.imshow("Annotated Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
