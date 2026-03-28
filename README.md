# Multimodal Deepfake Forensic Detection System

An advanced, production-ready AI forensic platform designed to combat highly sophisticated generative media, including AI-generated images, deepfake videos, voice cloning, and synthetically manipulated social media links. 

The system leverages a multi-microservice architecture to process massive spatial, temporal, and spectral tensor payloads concurrently, unifying them via a deterministic Cross-Modal Consensus Hub to detect zero-day digital manipulation.

---

## ⚡ Core Forensic Modalities

### 1. 🖼️ Spatial Image Forensics
A dedicated, isolated PyTorch microservice executing a 5-layer analysis pipeline:
- **Error Level Analysis (ELA):** Maps compression variances and identifies localized patching.
- **EfficientNet-B0 (`ai_detector_best.pth`):** Evaluates macro-structural artifacts caused by generative scaling.
- **ResNet-50 (`manipulation_detector_best.pth`):** Scans the stacked ELA tensor for splicing anomalies.
- **CLIP ViT-L/14 Context:** Verifies semantic alignment of image manifolds against latent textual cues.

### 2. 🎞️ Temporal Video Forensics
- **Precise Frame Extraction:** Dynamically calculates native container framerates and extracts exactly 3 isolated frames per second from the multimedia stream.
- **Synchronized Neural Tracking:** Bypasses conventional APIs by feeding the extracted frames directly into the shared local custom weights (`ai_detector_best` & `manipulation_detector_best`).
- **Temporal Optical Flow:** Computes dense Farneback optical matrices across continuous frames, tracking unnatural acceleration vectors symptomatic of deepfakes.
- **Bidirectional Synthesis Scanner:** Parses boundaries for ghosting artifacts caused by frame-interpolation architectures (RIFE/DAIN).

### 3. 🎙️ Acoustic Bio-Signal Forensics (6-Phase Pipeline)
- **Spectral Regularity:** Calculates MFCC arrays from Mel-Spectrograms to detect metronomic HiFi-GAN vocoding.
- **Sub-vocal Bio-Signals:** Scans temporal structures for F0 pitch jitter and natural inhalation/exhalation pauses.
- **Nexus-AudioForge-v2:** Ingests the waveform into a highly specialized `wav2vec2-base-960h` acoustic transformer. 
- **Speaker Cosine Distance:** Prevents voice-cloning by calculating identity similarity vectors across segmented audio.

### 4. 🌐 Headless Link Analysis
- **Anti-Bot Proxy Router:** Silently bypasses CDNs using headless browser tunneling.
- **Memory Demultiplexing:** Downloads and demultiplexes standard payload streams. Uses FFmpeg in-memory piping to split integrated MP4 structures into discrete RGB video tensors and PCM audio codecs.

### 5. 🧩 Cross-Modal Consensus Fusion
- **Lip-Sync Desynchronization (`calculate_lipsync_deviation`):** Mathematically traces visual kinetic text boundaries versus auditory spectral spikes.
- **Semantic Consistency (`verify_semantic_alignment`):** Ingests and enforces prompt consistency via mathematically calculated latent alignment algorithms (Cosine Similarity).

---

## 🧠 Locally Trained Custom Models
To avoid the network latency of cloud APIs and ensure absolute privacy, the system relies on proprietary local `.pth` neural weight checkpoints heavily fine-tuned specifically for this platform.

1. **`ai_detector_best.pth` (EfficientNet-B0 Baseline):** 
   * **Domain:** Spatial Image & Extracted Video Frames.
   * **Purpose:** Custom-trained on 40,000+ generative images (MidJourney v6, Stable Diffusion XL, DALL-E) to localize global synthetic noise profiles. It captures non-organic pixel relationships and voxel rendering artifacts invisible to the human eye.
   
2. **`manipulation_detector_best.pth` (ResNet-50 Patch Analyzer):** 
   * **Domain:** Spatial Image & Extracted Video Frames.
   * **Purpose:** Acts on an Error Level Analysis (ELA) sub-map. Custom-trained strictly on localized splicing attacks (e.g., face-swapping, object removal). It scans specifically for hard-edge mathematical inconsistencies on boundaries rather than global image generation.

3. **`Nexus-AudioForge-v2.pth` (wav2vec2-base-960h Transformer):** 
   * **Domain:** Vocal Cloning Forensics.
   * **Purpose:** A heavily fine-tuned iteration of an acoustic transformer trained to detect deepfake TTS vocoders (e.g., ElevenLabs, Bark). It tracks micro-jitter inside high-frequency phonemes.

---

## 🏗️ System Architecture 

This system employs a decoupled, multi-tier architecture to securely parallelize heavy tensor operations.

* **Frontend Client:** React 19 + TypeScript + Vite + Tailwind CSS. Utilizes `framer-motion` for complex continuous 50-second terminal logging telemetry.
* **Core API Gateway (Port 8000):** FastAPI server acting as the primary hub routing all Audio, Video, Link, and Fusion operations.
* **Image Microservice (Port 8002):** A wholly isolated FastAPI instance dedicated solely to operating the spatial 5-layer pipeline and PyTorch memory allocations.
* **Unified Model Weights (`/models`):** All custom trained `.pth` topologies are localized on-device to ensure privacy and circumvent external API latency.

---

## 💻 Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, PyTorch, Transformers (HuggingFace), OpenCV, Librosa, FFmpeg 
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Zustand, Framer Motion
- **Database:** Local forensic telemetry `db.json` logging (SQLite-scalable)

---

## 🚀 Installation & Setup

### 1. Requirements
Ensure you have the following installed on your machine:
- Node.js (v18+)
- Python (3.10+) 
- FFmpeg (added to system PATH)
- Conda or virtual environment (recommended)

### 2. Clone the Repository
```bash
git clone https://github.com/your-org/multimodal-fake-detector.git
cd multimodal-fake-detector
```

### 3. Setup Python Backend Environment
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install strictly enforced AI dependencies
pip install -r requirements.txt
```

### 4. Set Up Model Directory
Ensure your custom trained weights are located in the unified `./models` directory:
- `models/ai_detector_best.pth`
- `models/manipulation_detector_best.pth`

### 5. Setup React Frontend
```bash
cd frontend
npm install
cd ..
```

---

## ⚙️ Running the Application

To achieve fully optimized local inference, you will need to start three distinct services simultaneously.

### Terminal 1: Core API Gateway
This server handles incoming forensic streams, the Audio pipeline, the Video extraction logic, and multi-fusion data aggregation.
```bash
.\venv\Scripts\activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Image Microservice 
This specialized microservice spins up PyTorch solely to process the ELA Spatial pipeline safely outside the main thread.
```bash
.\venv\Scripts\activate
uvicorn image_analysis.api.main:app --host 0.0.0.0 --port 8002 --reload
```

### Terminal 3: Vite Frontend Client
Initializes the React user interface.
```bash
cd frontend
npm run dev
```

The application is now accessible locally. Proceed to [http://localhost:5173](http://localhost:5173).

---

## 🔐 Licensing
Proprietary Forensic Software. Use for academic or verified hackathon/demonstration purposes only. Do not scale for public production without establishing load-balancing on the internal PyTorch tensor allocation clusters.
