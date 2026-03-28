import cv2
import numpy as np

def detect_hand_anomaly(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return "Image not loaded"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 500:
        return "Possible manipulation or anomaly detected"

    return "No obvious anomaly detected"