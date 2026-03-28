"""
setup.py — Package configuration for the Deepfake Audio Detection System.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="deepfake-audio-detector",
    version="1.0.0",
    description=(
        "Production-grade deepfake audio detection via signal forensics. "
        "No ML training required."
    ),
    long_description=(
        "A complete audio forensics pipeline that detects AI-generated (TTS), "
        "voice-cloned, and spliced/edited audio using seven independent forensic signals: "
        "spectral smoothness, phase continuity, noise floor stationarity, pitch micro-variation, "
        "breathing detection, compression fingerprinting, and speaker embedding drift."
    ),
    author="Deepfake Detector",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deepfake-detect=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "deepfake", "audio", "detection", "forensics", "TTS",
        "voice-clone", "anti-spoofing", "signal-processing",
    ],
)
