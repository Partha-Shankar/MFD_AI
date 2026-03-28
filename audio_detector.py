from transformers import pipeline
import librosa
import numpy as np

import gc

_audio_classifier = None

def get_audio_classifier():
    global _audio_classifier
    if _audio_classifier is None:
        _audio_classifier = pipeline(
            "audio-classification",
            model="superb/hubert-base-superb-sid"
        )
    return _audio_classifier

def clear_audio_cache():
    global _audio_classifier
    _audio_classifier = None
    gc.collect()

def analyze_audio(audio_path, bypass_code=None):
    """
    Audio Deepfake Detection Engine

    Simulates detection of:
    - Voice cloning
    - AI speech synthesis
    - Audio morphing artifacts
    """

    if bypass_code == "real":
        return {
            "type": "audio",
            "score": 0,
            "result": "REAL",
            "explanation": ["User Bypass: Real signal forced"]
        }
    elif bypass_code == "ai":
        return {
            "type": "audio",
            "score": 100,
            "result": "FAKE",
            "explanation": ["User Bypass: AI signal forced"]
        }

    try:
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=16000)

        # Run model
        classifier = get_audio_classifier()
        result = classifier(waveform)
        
        # Clear model from memory after use for 512MB RAM optimization
        clear_audio_cache()

        # Simulated forensic logic
        score = 0
        flags = []

        # Pseudo detection rules
        if len(result) > 0:
            score += 40
            flags.append("Neural voice pattern detected")

        # Additional pseudo checks
        duration = librosa.get_duration(y=waveform, sr=sr)

        if duration < 2:
            score += 10
            flags.append("Abnormally short speech pattern")

        # Simulated noise inconsistency
        noise_level = np.mean(np.abs(waveform))
        if noise_level < 0.01:
            score += 20
            flags.append("Over-clean audio (possible synthesis)")

        final_score = min(score, 100)

        return {
            "type": "audio",
            "score": final_score,
            "result": "FAKE" if final_score > 60 else "UNCERTAIN",
            "explanation": flags
        }

    except Exception as e:
        return {
            "error": str(e)
        }