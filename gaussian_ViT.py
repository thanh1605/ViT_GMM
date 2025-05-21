import cv2
import numpy as np
from moviepy import VideoFileClip, ImageSequenceClip
from transformers import ViTImageProcessor, ViTModel
from sklearn.mixture import GaussianMixture
from PIL import Image
import torch

# Bước 1: Load video & extract frame
clip = VideoFileClip("video_cat.mp4")
frames = [frame for frame in clip.iter_frames()]
print(f"Đã trích xuất {len(frames)} frame từ video.")

# Bước 2: Phát hiện chuyển động (background subtraction)
fgbg = cv2.createBackgroundSubtractorMOG2()
boxes = []
for frame in frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    fgmask = fgbg.apply(frame_bgr)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 200:
            x, y, w, h = cv2.boundingRect(contour)
            frame_boxes.append([x, y, x + w, y + h])
    boxes.append(frame_boxes)

print("Tổng số bounding boxes:", sum(len(b) for b in boxes))

# Bước 3: Load mô hình ViT
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Bước 4: Hàm trích đặc trưng ViT
def extract_vehicle_features(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    cropped_image = frame[y1:y2, x1:x2]
    if cropped_image.size == 0:
        return None
    try:
        pil_image = Image.fromarray(cropped_image).resize((224, 224))
        inputs = vit_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Vector [CLS]
    except Exception as e:
        print("Lỗi khi trích đặc trưng:", e)
        return None

# Bước 5: Trích đặc trưng cho tất cả các box
vehicle_features = []
for frame, frame_boxes in zip(frames, boxes):
    frame_features = [extract_vehicle_features(frame, box) for box in frame_boxes]
    vehicle_features.append([f for f in frame_features if f is not None])

# Bước 6: Gom tất cả đặc trưng để GMM clustering
flat_features = [f for frame_feats in vehicle_features for f in frame_feats if f is not None]

if not flat_features:
    print("Không có đặc trưng nào được trích xuất.")
    exit()

print("Số đặc trưng được trích xuất:", len(flat_features))

# Bước 7: GMM clustering
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(flat_features)

# Gán label lại cho từng box trong từng frame
cluster_labels = []
for frame_feats in vehicle_features:
    if frame_feats:
        labels = gmm.predict(frame_feats)
    else:
        labels = []
    cluster_labels.append(labels)

# Bước 8: Vẽ box và xuất video
output_frames = []
for i, (frame, frame_boxes, labels) in enumerate(zip(frames, boxes, cluster_labels)):
    if len(frame_boxes) != len(labels):
        print(f"Frame {i}: Không khớp số box và số label -> bỏ qua")
        continue
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for box, label in zip(frame_boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Cluster {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    output_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

# Bước 9: Xuất video
if output_frames:
    final_clip = ImageSequenceClip(output_frames, fps=clip.fps)
    final_clip.write_videofile("video_cat_gmm.mp4")
    print("Video đã được xuất ra: video_cat_gmm.mp4")
else:
    print("Không có frame nào để xuất video.")
