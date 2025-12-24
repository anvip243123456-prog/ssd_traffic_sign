# -*- coding: utf-8 -*-
import os
import cv2
import torch
from torchvision import transforms
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.ops import nms

# -----------------------------
# CONFIG
# -----------------------------
checkpoint_path = "checkpoints/ssd300_vietnamese_traffic_signs_epoch_150.pth"
classes_file = "classes_vie.txt"
video_path = "test.mp4"         # video đầu vào
output_video_path = "results/video_output.mp4"  # video đầu ra
input_size = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_threshold = 0.3
nms_iou_threshold = 0.45

os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

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
    for param in model.backbone.parameters():
        param.requires_grad = False

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
# VIDEO PREDICTION
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

# Lấy thông số video
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    boxes = outputs["boxes"].cpu()
    labels = outputs["labels"].cpu()
    scores = outputs["scores"].cpu()

    keep = scores >= score_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    keep = nms(boxes, scores, nms_iou_threshold)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    scale_w, scale_h = w / input_size, h / input_size
    boxes = boxes * torch.tensor([scale_w, scale_h, scale_w, scale_h])

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.int().numpy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{classes[label-1]}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    out.write(frame)
    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out.release()
print(f"✅ Video saved: {output_video_path}")
