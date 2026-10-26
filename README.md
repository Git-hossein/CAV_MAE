# CAV-MAE & CAV-MAE Sync Feature Extraction Pipeline

Developed as part of my B.Sc. work, this repository provides efficient, batched data pipelines to extract multimodal audio and video embeddings using **CAV-MAE** and **CAV-MAE Sync** architectures.

---

## Prerequisites & Setup

Because this project relies on external model repositories, you must set up the project structure and checkpoints before running the scripts.

### 1. Clone Required Model Repositories
Ensure the following repositories are cloned inside your project root:

    # Clone standard CAV-MAE repo
    git clone https://github.com/YuanGongND/cav-mae.git ./cav_mae

    # Clone CAV-MAE Sync repo
    git clone https://github.com/Git-hossein/cav-mae-sync.git ./cav-mae-sync

### 2. Download Model Weights
Place your trained checkpoint `.pth` files in the root directory. As an example:
* `audio_model.21.pth` (Standard CAV-MAE checkpoint)
* `cav_mae_sync.pth` (CAV-MAE Sync checkpoint)

### 3. Installation
Install the necessary python libraries:

    pip install -r requirments.txt

---

## Pipeline Overview

### Audio Extraction (`extract_audio_frames_batched`)
* **Input:** A folder containing raw `.wav` audio files.
* **Process:** Resamples audio to 16kHz, computes 128-bin Kaldi Mel-filterbanks, pads/truncates to target sequence lengths, and runs batched forward passes through the audio transformer blocks.
* **Output:** Individual `.npy` embedding vectors saved per audio clip.

### Video Extraction (`extract_video_embeddings_batched`)
* **Input:** A parent folder containing subdirectories of video frames (`.jpg` / `.jpeg`).
* **Process:** Uniformly samples a target number of frames across time, applies CenterCrop/Resize/Normalization transforms, and extracts spatial-temporal token representations.
* **Output:** Clip-level and frame-level `.npy` embedding matrices.
