import numpy as np
import torch
import torchaudio
from pathlib import Path
import torchvision.transforms as T
import torch.nn.functional as F
import sys
import os
import glob
import re
import soundfile as sf
from PIL import Image
import PIL

sys.path.append("./cav-mae-sync")
from src.models.cav_mae_sync import CAVMAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants from dataloader_sync & retrieval.py for VGGSound ---
VGGSOUND_MEAN = -5.081
VGGSOUND_STD = 4.4849
NUM_FRAMES = 16
TARGET_LENGTH = 416  # 4s temporal context window for the sync model
NUM_MEL_BINS = 128

def load_cav_mae_sync(
    checkpoint_path: str = "cav_mae_sync.pth",
    audio_length: int = TARGET_LENGTH,
    num_register_tokens: int = 8,
    device: torch.device = device
) -> CAVMAE:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = CAVMAE(
        audio_length=audio_length,
        modality_specific_depth=11,
        num_register_tokens=num_register_tokens,
        cls_token=True,
        total_frame=NUM_FRAMES,
        contrastive_heads=False  # <--- Set this to False
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    clean_sd = {}
    for k, v in checkpoint.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_sd[clean_key] = v

    msg = model.load_state_dict(clean_sd, strict=False)
    print(f"Loaded weights from {checkpoint_path}: {msg}")

    model = model.to(device)
    model.eval()
    return model


##################################
# AUDIO
##################################

def map_frame_to_spectrogram(frame_index: int, num_frames: int, spectrogram_length: int, target_length: int):
    """
    Directly ported from src/dataloader_sync.py
    Extracts the window centered at the given frame index.
    """
    frame_position = int(round(frame_index * spectrogram_length / num_frames))
    start = max(0, frame_position - target_length // 2)
    end = start + target_length
    
    if end > spectrogram_length:
        end = spectrogram_length
        start = max(0, end - target_length)
    
    return start, end


def wav_to_fbank(wav_path: str, target_length: int = 1024, num_mel_bins: int = NUM_MEL_BINS) -> torch.Tensor:
    """
    Extracts 1024-step Kaldi filterbank matching dataloader_sync.py
    """
    data, sr = sf.read(wav_path, dtype="float32")
    waveform = torch.from_numpy(data)

    # Ensure shape is [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.t()

    # Downmix to mono if multi-channel (Kaldi requires [1, time])
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if not already
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    # Subtract mean
    waveform = waveform - waveform.mean()

    # Kaldi fbank feature extraction
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10
    )

    # Pad or truncate to target_length (1024)
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[:target_length, :]

    return fbank


@torch.no_grad()
def extract_audio_embeddings(
    wav_path: str,
    model: CAVMAE,
    num_frames: int = NUM_FRAMES,
    target_length: int = TARGET_LENGTH,
    device: torch.device = device
) -> torch.Tensor:
    """
    Processes a .wav file and returns frame-level audio embeddings of shape [16, 768].
    """
    # 1. Compute full 1024-frame filterbank
    full_fbank = wav_to_fbank(wav_path, target_length=1024)

    # 2. Slice filterbank into num_frames temporal windows
    fbanks = []
    for frame_idx in range(num_frames):
        start, end = map_frame_to_spectrogram(
            frame_index=frame_idx,
            num_frames=num_frames,
            spectrogram_length=full_fbank.shape[0],
            target_length=target_length
        )
        slice_fbank = full_fbank[start:end, :]

        # Normalize with VGGSound stats
        slice_fbank = (slice_fbank - VGGSOUND_MEAN) / VGGSOUND_STD
        fbanks.append(slice_fbank)

    # Shape: [16, 416, 128]
    audio_input = torch.stack(fbanks).to(device)

    # 3. Create dummy video input to satisfy the joint forward_feat signature
    # Shape: [16, 3, 224, 224]
    dummy_video = torch.zeros((num_frames, 3, 224, 224), device=device)

    # 4. Forward through the network
    with torch.amp.autocast(device_type=device.type):
        # forward_feat signature with cls_token=True returns: (ca, cv, cls_a, cls_v)
        tokens_audio, tokens_video, cls_audio, cls_video = model.forward_feat(audio_input, dummy_video)

        # L2-normalize across embedding dimension
        audio_emb = torch.nn.functional.normalize(cls_audio, dim=-1)

    # Shape: [16, 768] on CPU
    return audio_emb.cpu()







##################################
# VIDEO
##################################

image_preprocess = T.Compose([
    T.Resize(224, interpolation=PIL.Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.4850, 0.4560, 0.4060],
        std=[0.2290, 0.2240, 0.2250]
    )
])


def load_and_sample_frames(frame_dir: str, num_frames: int = 16) -> torch.Tensor:
    # Match all jpg frames and sort directly
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))

    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames matching 'frame_*.jpg' found in {frame_dir}")

    total_found = len(frame_paths)
    if total_found < num_frames:
        raise ValueError(f"Found {total_found} frames in {frame_dir}, expected at least {num_frames}.")

    # If you extracted exactly 16 frames, this takes all 16.
    # If you extracted more (e.g. 10 fps -> 100 frames), this samples 16 uniformly.
    indices = np.linspace(0, total_found - 1, num_frames, dtype=int)
    selected_paths = [frame_paths[idx] for idx in indices]

    tensors = []
    for path in selected_paths:
        with Image.open(path) as img:
            tensors.append(image_preprocess(img.convert("RGB")))

    return torch.stack(tensors)  # [16, 3, 224, 224]


@torch.no_grad()
def extract_video_embeddings(
    frame_dir: str,
    model: CAVMAE,
    num_frames: int = 16,
    target_audio_len: int = 416,
    device: torch.device = device
) -> torch.Tensor:
    """
    Processes a directory of video frames and returns frame-level 
    visual embeddings of shape [16, 768].
    """
    # 1. Load and transform the 16 video frames -> [16, 3, 224, 224]
    video_input = load_and_sample_frames(frame_dir, num_frames=num_frames).to(device)

    # 2. Create dummy audio input to satisfy forward_feat signature
    # Shape: [16, target_audio_len, 128]
    dummy_audio = torch.zeros((num_frames, target_audio_len, 128), device=device)

    # 3. Forward pass
    with torch.amp.autocast(device_type=device.type):
        # forward_feat returns: (ca, cv, cls_a, cls_v)
        tokens_audio, tokens_video, cls_audio, cls_video = model.forward_feat(dummy_audio, video_input)

        # L2-normalize visual CLS embeddings across the 768 dim
        video_emb = torch.nn.functional.normalize(cls_video, dim=-1)

    # Output shape: [16, 768] on CPU
    return video_emb.cpu()




if __name__ == "__main__":
    # Quick sanity test
    model = load_cav_mae_sync("./cav_mae_sync.pth")
    print("Model ready.")
    AUD_DIR = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/audios"
    VID_DIR = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/frames"
    # You can test with a sample file:
    other_cand = ["0Sam3gPcYLk", "2myGIZCgZ2g", "-0HxntVOLW4"]

    aud1_emb = extract_audio_embeddings(f"{AUD_DIR}/0Sam3gPcYLk.wav", model)
    aud2_emb = extract_audio_embeddings(f"{AUD_DIR}/2myGIZCgZ2g.wav", model)
    aud9_emb = extract_audio_embeddings(f"{AUD_DIR}/-0HxntVOLW4.wav", model)

    aud_emb = extract_audio_embeddings(f"{AUD_DIR}/-0aiCtGjNXE.wav", model)
    print("Extracted Audio Embedding Shape:", aud_emb.shape)  # Expected: torch.Size([16, 768])
    vid_emb = extract_video_embeddings(f"{VID_DIR}/-0aiCtGjNXE", model)
    print("Extracted Audio Embedding Shape:", vid_emb.shape)  # Expected: torch.Size([16, 768])

    for aud in [aud1_emb, aud2_emb, aud9_emb, aud_emb]:

        sim_matrix = torch.mm(vid_emb, aud.t())
        diagonal_sims = torch.diag(sim_matrix)

        # 3. Compute summary statistics
        diag_mean = diagonal_sims.mean().item()
        diag_max = diagonal_sims.max().item()

        # 4. Off-diagonal mean (misaligned moments from the same video)
        mask = torch.eye(16, dtype=torch.bool)
        off_diag_mean = sim_matrix[~mask].mean().item()

        print("\n--- Temporal Alignment Sanity Check ---")
        print(f"Similarity Matrix Shape: {sim_matrix.shape}")
        print(f"\nDiagonal Mean Score: {diag_mean:.4f}")
        print(f"Diagonal Max Score:  {diag_max:.4f}")
        print(f"Off-Diagonal Mean:   {off_diag_mean:.4f}")
        print(f"Sync Margin (Diag - OffDiag): {diag_mean - off_diag_mean:.4f}")