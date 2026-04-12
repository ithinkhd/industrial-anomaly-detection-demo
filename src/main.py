import cv2
import os
from detect import detect_and_crop
from anomaly import build_memory, compute_anomaly_score

# Load normal images
normal_images = []
for file in os.listdir("C:\\Anomaly Detection\\industrial-anomaly-detection-demo\\data\\normal"):
    path = os.path.join("C:\\Anomaly Detection\\industrial-anomaly-detection-demo\\data\\normal", file)
    img = cv2.imread(path)
    if img is not None:
        normal_images.append(img)

build_memory(normal_images)

# Test images
for file in os.listdir("C:\\Anomaly Detection\\industrial-anomaly-detection-demo\\data\\test"):
    path = os.path.join("C:\\Anomaly Detection\\industrial-anomaly-detection-demo\\data\\test", file)

    result = detect_and_crop(path)

    if result is None:
        print(f"[WARNING] No detection for {path}")

        import cv2
        img = cv2.imread(path)

        # fallback → use full image
        cropped = img
        original = img
        box = (0, 0, img.shape[1], img.shape[0])
    else:
        cropped, original, box = result

    if cropped is None:
        continue

    score = compute_anomaly_score(cropped)

    x1, y1, x2, y2 = box

    label = "ANOMALY" if score > 0.3 else "NORMAL"

    cv2.rectangle(original, (x1, y1), (x2, y2),
                  (0, 0, 255) if label == "ANOMALY" else (0, 255, 0), 2)

    cv2.putText(original, f"{label} ({score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if label == "ANOMALY" else (0, 255, 0), 2)

    cv2.imshow("Result", original)
    cv2.waitKey(0)

cv2.destroyAllWindows()