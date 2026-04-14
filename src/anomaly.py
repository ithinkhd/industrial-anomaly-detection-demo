import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

memory_bank = []


# FEATURE EXTRACTION
def extract_feature(img):
    img = cv2.resize(img, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Texture feature (histogram)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Edge feature (defect sensitive)
    edges = cv2.Canny(gray, 50, 150)
    edge_feat = edges.flatten() / 255.0

    # Combine features
    feature = np.concatenate([hist, edge_feat])

    return feature

# BUILD MEMORY BANK
def build_memory(normal_images):
    global memory_bank
    memory_bank = [extract_feature(img) for img in normal_images]

# ANOMALY SCORE
def compute_anomaly_score(test_img):
    test_feature = extract_feature(test_img)

    similarities = [
        cosine_similarity([test_feature], [mem])[0][0]
        for mem in memory_bank
    ]

    # take TOP-K stability (PatchCore idea simplified)
    top_k = sorted(similarities, reverse=True)[:5]

    return 1 - np.mean(top_k)