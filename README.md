# Industrial Anomaly Detection (YOLO + Feature-Based)

## Overview
This project demonstrates a hybrid anomaly detection system inspired by industrial inspection workflows.

The pipeline combines:
- Object detection (YOLO)
- Feature extraction
- Memory bank comparison (PatchCore-inspired)

## Workflow
1. Detect object using YOLO
2. Crop detected region
3. Extract feature vector
4. Compare with normal memory bank
5. Compute anomaly score

## Technologies
- Python
- OpenCV
- YOLO (Ultralytics)
- Scikit-learn

## Results
The system is able to:
- Detect objects in images
- Identify anomalies based on feature similarity
- Display bounding boxes with anomaly scores

## Disclaimer
This project is a simplified demonstration for portfolio purposes.
No proprietary or confidential data is used.

## How to Run
```bash
pip install -r requirements.txt
python src/main.py