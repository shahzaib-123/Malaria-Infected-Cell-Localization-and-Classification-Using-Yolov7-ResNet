import torch
import os
import sys
sys.path.append('/workspaces/codespaces-blank/yolov7')
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression
from yolov7.utils.datasets import letterbox
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

def scale_boxes(img_shape, boxes, ratio, padding):
    """
    Scales bounding boxes back to the original image size after padding and resizing.
    """
    height, width = img_shape
    r, p = ratio, padding
    
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / r[1]  # scale x-coordinates
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / r[0]  # scale y-coordinates

    boxes[:, [0, 2]] -= p[1]  # adjust x-coordinates by left padding
    boxes[:, [1, 3]] -= p[0]  # adjust y-coordinates by top padding

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=height)

    return boxes

def detect_objects(image, model, device, conf_thres=0.5, iou_thres=0.45):
    img, ratio, padding = letterbox(image, new_shape=640, stride=32, auto=True)
    
    print(f"Ratio (scaling factor): {ratio}")
    print(f"Padding (left, top): {padding}")
    
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and transpose to CxHxW
    img = np.ascontiguousarray(img)
    
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0  # Normalize to 0-1
    img_tensor = img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor

    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    
    if pred is not None and len(pred):
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Scale the bounding boxes back to the original size
                det[:, :4] = scale_boxes(image.shape[:2], det[:, :4], ratio, padding)  # Ensure scaling is done correctly
    return pred

yolo_model_path = "/workspaces/codespaces-blank/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = attempt_load(yolo_model_path, map_location=device)
yolo_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 170)),
    transforms.ToTensor()
])

class_names = {0: 'difficult', 1: 'gametocyte', 2: 'leukocyte', 3: 'rbc', 
               4: 'ring', 5: 'schizont', 6: 'trophozoite'}

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 7)  # Replace the last layer
model = model.to(device)
checkpoint_path = "/workspaces/codespaces-blank/cell_classification_resnet.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

image_path = "/workspaces/codespaces-blank/c691c8ef-d3c7-47b6-a941-f6a1089f21a9.png"
original_image = cv2.imread(image_path)
detections = detect_objects(original_image, yolo_model, device)

cropped_images = []
bboxes = []

for det in detections:  # Loop through detections
    if det is not None and len(det):
        for *xyxy, conf, cls in det:  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)
            cropped = original_image[y1:y2, x1:x2]  # Crop region
            cropped_images.append(cropped)
            bboxes.append((x1, y1, x2, y2))

model.eval()
predicted_labels = []

with torch.no_grad():
    for cropped in cropped_images:
        input_tensor = preprocess(cropped).unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_labels.append(predicted_class.item())

annotated_image = original_image.copy()
for bbox, label in zip(bboxes, predicted_labels):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw bounding box
    cv2.putText(
        annotated_image,
        class_names[label],
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0, 0),
        2,
    )

cv2.imwrite("annotated_image3.jpg", annotated_image)
