from image_detector import detect_fake_image
from video_detector import detect_fake_video
from anomaly_detector import detect_hand_anomaly

import os

def analyze_media(file):

    ext = file.split(".")[-1]

    if ext in ["jpg","png","jpeg"]:

        print("Running Image AI detector...")
        result1 = detect_fake_image(file)

        print("Running Hand anomaly detection...")
        result2 = detect_hand_anomaly(file)

        print("IMAGE RESULT:", result1)
        print("ANOMALY RESULT:", result2)

    elif ext in ["mp4","avi","mov"]:

        print("Running Video Deepfake detector...")
        result = detect_fake_video(file)

        print(result)
    
    elif is_audio(file):
        return analyze_audio(file)

    else:
        print("Unsupported file")

if __name__ == "__main__":

    file = input("Enter file path:")

    analyze_media(file)