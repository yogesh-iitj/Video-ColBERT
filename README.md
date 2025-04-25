
# Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval. [original paper](https://arxiv.org/pdf/2503.19009v1)


This repository implements Video-ColBERT, a contextualized late interaction model for text-to-video retrieval. Video-ColBERT performs fine-grained token-wise interactions between text queries and video content.

## Overview

Video-ColBERT introduces a simple and efficient mechanism for fine-grained similarity assessment between text queries and videos through:

1. **Dual token-wise interaction** - Performs MeanMaxSim (MMS) on both static frame features and temporally contextualized video features
2. **Query and visual expansion tokens** - Provides learnable expansion to improve matching for abstract queries
3. **Temporal modeling** - Effectively captures temporal relationships between frames

## Installation

```bash
# Clone the repository
https://github.com/yogesh-iitj/Video-ColBERT
cd video-colbert

# Install dependencies
pip install torch torchvision numpy pillow einops decord
pip install git+https://github.com/openai/CLIP.git
```

## Quick Start

```python
from video_colbert import VideoColBERT, load_video_frames
import torch

# Initialize model
model = VideoColBERT(backbone_name="ViT-B/32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load video frames from a file
video_path = "sample_video.mp4"
frames = load_video_frames(video_path, num_frames=12).to(device)

# Process query and compute similarity
query = "a police officer drives his car onto a grassy field"
with torch.no_grad():
    # Get query and video representations
    query_features = model.get_query_representation(query)
    frame_features, video_features = model.get_video_representations(frames)
    
    # Compute similarity
    similarity, frame_sim, video_sim = model.compute_similarity(
        query_features.unsqueeze(0),
        frame_features.unsqueeze(0),
        video_features.unsqueeze(0)
    )

print(f"Query: {query}")
print(f"Overall similarity: {similarity.item():.4f}")
print(f"Frame-level: {frame_sim.item():.4f}, Video-level: {video_sim.item():.4f}")
```

## Testing with Random Inputs

For quick testing without video files:

```bash
python example.py
```

This script demonstrates the model with random inputs, showing how similarity matrices are computed between text queries and video frames.

## How It Works

1. **Text Encoding**: The query is encoded by a text encoder (CLIP)
2. **Frame Encoding**: Video frames are encoded independently with an image encoder
3. **Temporal Encoding**: Frame features are processed by a temporal transformer
4. **Dual Similarity**: The model performs dual token-wise similarity calculation:
   - Each query token scans all frame features and keeps the maximum (spatial similarity)
   - Each query token scans all temporally contextualized features and keeps the maximum (temporal similarity)
   - Final score combines both spatial and temporal similarities

This approach effectively captures both static visual concepts and dynamic temporal relationships for better text-to-video retrieval.

## Paper

For more details, please refer to the [original paper](https://arxiv.org/pdf/2503.19009v1).
