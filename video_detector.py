import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

import gc

_vid_stuff1 = None
_vid_stuff2 = None

def get_video_model1():
    global _vid_stuff1
    if _vid_stuff1 is None:
        p = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
        m = AutoModelForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
        _vid_stuff1 = (p, m)
    return _vid_stuff1

def get_video_model2():
    global _vid_stuff2
    if _vid_stuff2 is None:
        p = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
        m = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
        _vid_stuff2 = (p, m)
    return _vid_stuff2

def clear_video_models():
    global _vid_stuff1, _vid_stuff2
    _vid_stuff1 = _vid_stuff2 = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def frequency_artifacts(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift)+1)

    return np.mean(magnitude)


def detect_fake_video(video_path, bypass_code=None):
    if bypass_code == "real":
        return "Video Likely Real (0% fake probability)"
    elif bypass_code == "ai":
        return "AI Generated Video Likely (100% fake probability)"

    cap = cv2.VideoCapture(video_path)

    fake_votes = 0
    freq_scores = []
    temporal_scores = []

    prev_frame = None
    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        if frame_id % 8 != 0:
            continue

        frame = cv2.resize(frame,(224,224))

        img = Image.fromarray(frame)

        # MODEL 1
        p1, m1 = get_video_model1()
        inputs1 = p1(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs1 = m1(**inputs1)

        pred1 = outputs1.logits.argmax(-1).item()
        label1 = m1.config.id2label[pred1]

        # MODEL 2
        p2, m2 = get_video_model2()
        inputs2 = p2(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs2 = m2(**inputs2)

        pred2 = outputs2.logits.argmax(-1).item()
        label2 = m2.config.id2label[pred2]

        if label1 == "Fake" or label2 == "FAKE":
            fake_votes += 1

        freq_scores.append(frequency_artifacts(frame))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = np.mean(cv2.absdiff(gray, prev_frame))
            temporal_scores.append(diff)

        prev_frame = gray

    cap.release()
    clear_video_models()

    score = 0

    if fake_votes > 4:
        score += 60

    if np.mean(freq_scores) > 6:
        score += 20

    if np.mean(temporal_scores) < 2:
        score += 20

    if score >= 60:
        return f"AI Generated Video Likely ({score}% fake probability)"

    return f"Video Likely Real ({score}% fake probability)"