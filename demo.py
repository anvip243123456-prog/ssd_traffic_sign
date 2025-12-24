# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.ops import nms

# -----------------------------
# CONFIG
# -----------------------------
checkpoint_path = "checkpoints/ssd300_vietnamese_traffic_signs_epoch_150.pth"
classes_file = "classes_vie.txt"
image_folder = "vietnamese-traffic-signs/archive/images"
results_folder = "results"
input_size = 300   # SSD300 input size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_threshold = 0.3
nms_iou_threshold = 0.45
max_images = 300  # chỉ xử lý 300 ảnh đầu tiên

os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# LOAD CLASSES
# -----------------------------
with open(classes_file, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

num_classes = len(classes) + 1  # +1 for background
print(f"Detected {len(classes)} classes (excluding background)")

# -----------------------------
# CREATE MODEL
# -----------------------------
def create_model(num_classes, size=300):
    model = torchvision.models.detection.ssd300_vgg16(weights=None)
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Replace classification head
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
model = create_model(num_classes, size=input_size)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print(f"✅ Loaded checkpoint: {checkpoint_path}")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    # Get boxes, labels, scores
    boxes = outputs["boxes"].cpu()
    labels = outputs["labels"].cpu()
    scores = outputs["scores"].cpu()

    # Filter by score threshold
    keep = scores >= score_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # Apply NMS
    keep = nms(boxes, scores, nms_iou_threshold)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # Rescale boxes to original image size
    scale_w, scale_h = w / input_size, h / input_size
    boxes = boxes * torch.tensor([scale_w, scale_h, scale_w, scale_h])

    return boxes.numpy(), labels.numpy(), scores.numpy(), img

# -----------------------------
# RUN PREDICTIONS ON FIRST 300 IMAGES
# -----------------------------
# Sắp xếp file theo tên để lấy 300 ảnh đầu tiên
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])

for idx, file_name in enumerate(image_files):
    if idx >= max_images:
        break

    img_path = os.path.join(image_folder, file_name)
    boxes, labels, scores, img = predict_image(img_path)

    # Draw results
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        text = f"{classes[label-1]}: {score:.2f}"  # -1 vì label 0 là background
        cv2.putText(img, text, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    out_path = os.path.join(results_folder, f"{idx+1:04d}.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved result: {out_path}")

print(f"✅ Processed first {max_images} images!")
