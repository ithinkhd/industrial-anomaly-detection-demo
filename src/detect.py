from ultralytics import YOLO
import cv2

# for more specific items/product, can consider using trained model (best.pt)
model = YOLO("C:\\Anomaly Detection\\runs\\detect\\yolo11n_anomaly3\\weights\\best.pt")  # Change Own PATH

def detect_and_crop(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None, None, None

    results = model(img)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        # fallback: use full image
        h, w = img.shape[:2]
        return img, img, (0, 0, w, h)

    x1, y1, x2, y2 = map(int, boxes.xyxy[0])

    cropped = img[y1:y2, x1:x2]

    return cropped, img, (x1, y1, x2, y2)