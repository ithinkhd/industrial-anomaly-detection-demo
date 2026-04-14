# 🔍 Industrial Anomaly Detection System (YOLO + PatchCore-Inspired)

## 📌 Overview
This project demonstrates an end-to-end industrial visual inspection pipeline for detecting surface defects on screws using a hybrid AI approach.

The system integrates:
- YOLO-based object detection for region localization
- Feature-based anomaly detection (PatchCore-inspired)
- Memory bank comparison using normal samples
- Visual heatmap and anomaly scoring

---

## ⚙️ System Pipeline
Input Image
↓
YOLO Object Detection
↓
Region Cropping
↓
Feature Extraction
↓
Memory Bank Comparison
↓
Anomaly Scoring
↓
Visualization (Bounding Box + Score)

---

## 🧠 Key Features
- Real-time object detection using YOLO
- Feature-based anomaly detection (no need for defect labels)
- Memory bank built from normal samples only
- Anomaly scoring using similarity distance
- Visual output with bounding boxes and confidence score

---

## 🛠️ Technologies Used
- Python
- OpenCV
- Ultralytics YOLO
- Scikit-learn
- NumPy

---

## 📊 Results

The system can:
- Detect screws in images
- Identify surface defects (scratches, anomalies, irregularities)
- Classify results as **NORMAL / ANOMALY**
- Display bounding boxes with anomaly scores

Example outputs:
- Green box → Normal screw
- Red box → Defective screw

---

## ⚠️ Limitations & Future Improvements

This project is a proof-of-concept system and has the following limitations:

- Standard bounding boxes may include background noise
- Performance depends on YOLO detection accuracy
- Feature-based anomaly detection is sensitive to lighting conditions

### Future improvements:
- Implementation of Oriented Bounding Boxes (OBB) for better alignment
- Use of deep feature extractors (ResNet / PatchCore full implementation)
- Improved dataset normalization and lighting correction
- Real-time industrial deployment optimization

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/main.py