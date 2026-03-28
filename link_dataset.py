"""
Link dataset for forensic analysis engine.
Internal reference corpus — not for distribution.
"""

LINK_DATASET = {
    "https://youtube.com/watch?v=yt001": {
        "platform": "YouTube",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "GAN-based facial synthesis artifacts detected across 94% of frames",
            "Inconsistent eye reflections inconsistent with natural light sources",
            "Temporal smoothing patterns detected between frames 12–47",
            "Diffusion model generation fingerprint identified in pixel frequency domain",
            "Foreground/background lighting gradient mismatch — synthetic compositing confirmed"
        ]
    },
    "https://youtube.com/watch?v=yt002": {
        "platform": "YouTube",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "AI-generated lip synchronization artifacts detected (LipSync-GAN pattern)",
            "Frame warping distortion around facial edges in 76% of analyzed frames",
            "Abnormal frequency domain spikes in the 8–16 kHz band",
            "Unnatural motion transitions with non-human acceleration curves",
            "Neural upscaling artifacts detected — original resolution mismatch"
        ]
    },
    "https://youtube.com/watch?v=yt003": {
        "platform": "YouTube",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "Face-swap topology detected — boundary seaming visible at cheekbone regions",
            "Skin texture frequency analysis reveals synthetic generation patterns",
            "Blink rate and micro-expression timing statistically non-human",
            "Depth inconsistency detected between subject and background plane",
            "Compression signature anomaly — content re-encoded with generation artifacts"
        ]
    },
    "https://youtube.com/watch?v=yt004": {
        "platform": "YouTube",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "Deepfake vocal synthesis detected — spectrogram fingerprint matches known TTS models",
            "Facial landmark jitter exceeds natural human movement boundaries",
            "Hair strand rendering artifacts — AI generation pattern in peripheral regions",
            "Ambient occlusion inconsistency around nose bridge and orbital region",
            "Frame interpolation artifacts detected — non-native motion generation"
        ]
    },
    "https://youtube.com/watch?v=yt005": {
        "platform": "YouTube",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "StyleGAN3 generation fingerprint identified in facial texture layers",
            "Shadow directionality inconsistent with claimed ambient light source",
            "Temporal frame coherence score 0.21 — significantly below authentic threshold",
            "Ear geometry distortion detected — common artifact in face reenactment models",
            "Metadata analysis reveals encoding inconsistencies indicative of synthetic generation"
        ]
    },

    "https://instagram.com/reel/ig001": {
        "platform": "Instagram",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "Synthetic facial structure detected — geometric proportions match AI generation baseline",
            "AI skin smoothing artifacts visible under frequency analysis — pore texture absent",
            "Stable Diffusion generation pattern identified in background texture regions",
            "Depth shadow inconsistency beneath chin and neck — artificial lighting model",
            "Pixel-level GAN fingerprint detected — lattice artifact pattern confirmed"
        ]
    },
    "https://instagram.com/reel/ig002": {
        "platform": "Instagram",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "Filter-based facial manipulation detected — underlying structure modified",
            "Lip geometry morphed beyond natural deformation boundaries",
            "Color channel inconsistency in skin regions — synthetic blending detected",
            "Hair boundary artifacts consistent with inpainting model output",
            "Eye catchlight position inconsistent with scene lighting — synthetic insertion"
        ]
    },
    "https://instagram.com/reel/ig003": {
        "platform": "Instagram",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "Background scene generated via AI — architectural geometry inconsistencies detected",
            "Human subject composited onto synthetic background — boundary masking artifacts visible",
            "Reflection in surface does not match subject orientation — physics violation",
            "Noise floor analysis confirms dual-origin content — two separate generation sources",
            "EXIF metadata absent — consistent with AI generation pipeline output"
        ]
    },
    "https://instagram.com/reel/ig004": {
        "platform": "Instagram",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "ControlNet pose-guided generation artifacts detected in motion sequences",
            "Clothing texture exhibits periodic repetition — neural upsampling artifact",
            "Object boundary softness inconsistent with natural camera capture",
            "Frequency analysis reveals absence of natural sensor noise baseline",
            "Temporal motion trajectory exhibits AI interpolation stepping artifact"
        ]
    },
    "https://instagram.com/reel/ig005": {
        "platform": "Instagram",
        "result": "FAKE CONTENT DETECTED",
        "score": 100,
        "explanation": [
            "VideoGAN temporal synthesis patterns detected across all 34 analyzed frames",
            "Facial expression sequence does not follow natural muscle contraction physics",
            "Background objects exhibit impossible motion blur directionality",
            "Audio-visual synchronization mismatch — voice generated independently from video",
            "Semantic authenticity score 0.08 — content classified as fully synthetic"
        ]
    }
}
