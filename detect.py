from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # lightweight model

def detect_and_crop(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # take first detected object
    x1, y1, x2, y2 = map(int, boxes.xyxy[0])
    cropped = img[y1:y2, x1:x2]

    return cropped, img, (x1, y1, x2, y2)