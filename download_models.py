import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

def download_and_rename_models():
    os.makedirs("./Nexus_Custom_Models", exist_ok=True)
    
    print("Downloading weights for Nexus-VisionGuard-v2...")
    # Downloading an open source model
    processor1 = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
    model1 = AutoModelForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
    
    # Saving it locally under our custom name
    processor1.save_pretrained("./Nexus_Custom_Models/Nexus-VisionGuard-v2")
    model1.save_pretrained("./Nexus_Custom_Models/Nexus-VisionGuard-v2")
    print("Nexus-VisionGuard-v2 saved to ./Nexus_Custom_Models/Nexus-VisionGuard-v2")

    print("Downloading weights for Nexus-AcoustoKinetix-v1...")
    # Downloading another open source model
    processor2 = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model2 = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    
    # Saving it locally under our custom name
    processor2.save_pretrained("./Nexus_Custom_Models/Nexus-AcoustoKinetix-v1")
    model2.save_pretrained("./Nexus_Custom_Models/Nexus-AcoustoKinetix-v1")
    print("Nexus-AcoustoKinetix-v1 saved to ./Nexus_Custom_Models/Nexus-AcoustoKinetix-v1")

if __name__ == "__main__":
    download_and_rename_models()
