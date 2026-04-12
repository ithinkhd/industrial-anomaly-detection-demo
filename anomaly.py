import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

memory_bank = []

def extract_feature(img):
    img = cv2.resize(img, (128, 128))
    feature = img.flatten() / 255.0
    return feature

def build_memory(normal_images):
    global memory_bank
    memory_bank = [extract_feature(img) for img in normal_images]

def compute_anomaly_score(test_img):
    test_feature = extract_feature(test_img)

    similarities = [
        cosine_similarity([test_feature], [mem])[0][0]
        for mem in memory_bank
    ]

    return 1 - max(similarities)  # higher = more anomaly