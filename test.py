from pathlib import Path
from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchaudio
import uvicorn
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

# Path to CAV-MAE model repository
sys.path.append("./cav_mae")
from src.models.cav_mae import CAVMAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_instance = None


# =====================================================================
# 1. MODEL INITIALIZATION
# =====================================================================
CAV_MAE_Scale_plusplus = "audio_model.21.pth"

def get_model(ckpt_path: str = None):
    """Loads and caches CAV-MAE model in memory."""
    global _model_instance
    if _model_instance is None:
        print(f"⏳ Loading CAV-MAE model on {device}...")
        _model_instance = CAVMAE(
            audio_length=1024,
            modality_specific_depth=11,
            norm_pix_loss=False,
            tr_pos=False,
        )

        if ckpt_path and Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = (
                checkpoint["model"] if "model" in checkpoint else checkpoint
            )

            # Strip 'module.' prefix from DataParallel
            clean_state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            msg = _model_instance.load_state_dict(
                clean_state_dict, strict=False
            )
            print(f" Loaded checkpoint weights from: {ckpt_path}")
            print(f"   Missing/Unexpected keys: {msg}")
        else:
            print("⚠️ Running with randomly initialized weights (No checkpoint).")

        _model_instance.to(device)
        _model_instance.eval()
        print(" CAV-MAE model loaded and ready.")
    return _model_instance


# =====================================================================
# 1.5 CUSTOM FORWARD PASS
# =====================================================================

def forward_audio_only(model: torch.nn.Module, a: torch.Tensor) -> torch.Tensor:
    """Executes ONLY the audio pipeline of CAV-MAE.

    Input `a`: [B, 1024, 128] Output: [B, 512, 768]
    """
    # 1. Patch projection & modality embeddings
    a = a.unsqueeze(1)
    a = a.transpose(2, 3)
    a = model.patch_embed_a(a)
    a = a + model.pos_embed_a
    a = a + model.modality_a

    # 2. Audio-specific Transformer blocks
    for blk in model.blocks_a:
        a = blk(a)

    # 3. Unified blocks routed to audio normalization
    for blk in model.blocks_u:
        a = blk(a, "a")

    a = model.norm_a(a)
    return a


def forward_video_only(model: torch.nn.Module, v: torch.Tensor) -> torch.Tensor:
    """Executes ONLY the video pipeline of CAV-MAE.

    Input `v`: [B, 3, 224, 224] Output: [B, 196, 768]
    """
    # 1. Patch projection & modality embeddings
    v = model.patch_embed_v(v)
    v = v + model.pos_embed_v
    v = v + model.modality_v

    # 2. Visual-specific Transformer blocks
    for blk in model.blocks_v:
        v = blk(v)

    # 3. Unified blocks routed to visual normalization
    for blk in model.blocks_u:
        v = blk(v, "v")

    v = model.norm_v(v)
    return v


# =====================================================================
# 2. AUDIO BATCH EXTRACTION FUNCTION
# =====================================================================

class AudioExtractionDataset(Dataset):

    def __init__(
        self,
        wav_paths: list,
        target_sample_rate: int = 16000,
        target_length: int = 1024,
        norm_mean: float = -5.081,
        norm_std: float = 4.4849,
    ):
        self.wav_paths = wav_paths
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]

        # 1. Load audio file
        data, sr = sf.read(str(wav_path), dtype="float32")
        waveform = torch.from_numpy(data)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
        else:
            waveform = waveform.t()  # [channels, samples]

        # 2. Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate
            )
            waveform = resampler(waveform)

        # 4. Zero-center waveform
        waveform = waveform - waveform.mean()

        # 5. 128-bin Mel Filterbank
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.target_sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )  # [time_frames, 128]

        # 6. Pad or truncate to target_length (1024)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            pad = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = pad(fbank)
        elif p < 0:
            fbank = fbank[: self.target_length, :]

        # 7. CAV-MAE AudioSet Normalization
        fbank = (fbank - self.norm_mean) / self.norm_std

        # Shape returned per item: [1024, 128]
        return fbank, str(wav_path)


def preprocess_audio(
    wav_path: str,
    target_sample_rate: int = 16000,
    target_length: int = 1024,
    norm_mean: float = -5.081,
    norm_std: float = 4.4849,
) -> torch.Tensor:
    """Preprocesses raw .wav audio into the [1, 1024, 128] tensor CAV-MAE expects."""
    data, sr = sf.read(wav_path, dtype="float32")
    waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, samples]
    else:
        waveform = waveform.t()  # [channels, samples]

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # CAV-MAE specific: zero-center the waveform
    waveform = waveform - waveform.mean()

    # 128 Mel Filterbanks
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=target_sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )  # [time_frames, 128]

    # Pad or truncate to target_length
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = pad(fbank)
    elif p < 0:
        fbank = fbank[:target_length, :]

    # Normalize with standard CAV-MAE AudioSet statistics (divided by std, NOT std * 2)
    fbank = (fbank - norm_mean) / norm_std

    # Shape: [1, 1024, 128]
    return fbank.unsqueeze(0)

def extract_audio_frames_batched(
    input_folder: str,
    output_folder: str,
    ckpt_path: str = CAV_MAE_Scale_plusplus,
    batch_size: int = 32,
    num_workers: int = 4,
    skip_existing: bool = True,
    l2_normalize: bool = True,
):
    """Batched extraction of audio embeddings with parallel CPU preprocessing."""
    model = get_model(ckpt_path).to(device)
    model.eval()

    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_wav_files = sorted(list(in_dir.glob("*.wav")))
    if not all_wav_files:
        print(f"⚠️ No .wav files found in {input_folder}")
        return {"processed": 0, "saved_to": str(out_dir)}

    # Filter out already extracted files upfront
    if skip_existing:
        wav_files = [
            p for p in all_wav_files if not (out_dir / f"{p.stem}.npy").exists()
        ]
        skipped_count = len(all_wav_files) - len(wav_files)
        if skipped_count > 0:
            print(f"⏭️ Skipping {skipped_count} already extracted files.")
    else:
        wav_files = all_wav_files

    if not wav_files:
        print("✅ All files already have embeddings. Nothing to process.")
        return {"processed": 0, "saved_to": str(out_dir)}

    print(
        f"▶️ Processing {len(wav_files)} files (Batch Size: {batch_size}, Workers: {num_workers})..."
    )

    # Initialize PyTorch Dataset & DataLoader
    dataset = AudioExtractionDataset(wav_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    saved_count = 0
    for batch_fbanks, batch_paths in loader:
        # batch_fbanks shape: [B, 1024, 128]
        batch_fbanks = batch_fbanks.to(device, non_blocking=True)

        with torch.no_grad():
            # 1. Batched forward pass through audio branch -> [B, 512, 768]
            audio_tokens = forward_audio_only(model, batch_fbanks)

            # 2. Mean-pool across 512 patch tokens -> [B, 768]
            audio_embs = audio_tokens.mean(dim=1)

            # 3. L2 normalization across embedding dimension
            if l2_normalize:
                audio_embs = F.normalize(audio_embs, dim=-1)

            # Move batch embeddings to CPU
            audio_embs = audio_embs.cpu().numpy()

        # 4. Save each vector individually as .npy
        for emb, path_str in zip(audio_embs, batch_paths):
            stem = Path(path_str).stem
            out_file = out_dir / f"{stem}.npy"
            np.save(str(out_file), emb)
            saved_count += 1

        print(f" Saved {saved_count}/{len(wav_files)} embeddings...", end="\r")

    print(
        f"\n✅ Finished! Saved {saved_count} audio embeddings to: {output_folder}"
    )
    return {"processed": saved_count, "saved_to": str(output_folder)}

# =====================================================================
# 3. VIDEO BATCH EXTRACTION FUNCTION
# =====================================================================

video_transform = T.Compose(
    [
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
        ),
    ]
)

class VideoFramesDataset(Dataset):
    def __init__(self, video_folders: list, num_frames: int = 10):
        self.video_folders = video_folders
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        folder_path = self.video_folders[idx]
        
        # Support both .jpg and .jpeg
        frame_paths = sorted(
            list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
        )
        total_frames = len(frame_paths)

        if total_frames == 0:
            raise ValueError(f"No frames found in: {folder_path}")

        # Uniform sampling across time
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for i in indices:
            with Image.open(frame_paths[i]) as img:
                frames.append(video_transform(img.convert("RGB")))

        # Shape returned per item: [num_frames, 3, 224, 224]
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, folder_path.name

def load_video_from_frame_folder(
    frame_dir: Path, num_frames: int = 5, device: torch.device = None
) -> torch.Tensor:
    """Samples `num_frames` uniformly from the folder and returns a normalized tensor [num_frames, 3, 224, 224]."""
    # Support both .jpg and .jpeg
    frame_paths = sorted(
        list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg"))
    )
    total_frames = len(frame_paths)

    if total_frames == 0:
        raise ValueError(f"No .jpg frames found in: {frame_dir}")

    # Uniform sampling across time
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        with Image.open(frame_paths[idx]) as img:
            frames.append(video_transform(img.convert("RGB")))

    # Shape: [num_frames, 3, 224, 224]
    video_tensor = torch.stack(frames, dim=0)
    if device is not None:
        video_tensor = video_tensor.to(device)
    return video_tensor

def extract_video_embeddings_batched(
    parent_frames_dir: str,
    output_folder_clip: str,
    output_folder_frame: str,
    ckpt_path: str = CAV_MAE_Scale_plusplus,
    num_frames: int = 10,
    batch_size: int = 8,
    num_workers: int = 4,
    skip_existing: bool = True,
    l2_normalize: bool = True,
):
    """Batched extraction of video embeddings with parallel CPU frame loading.

    Saves:
      - Clip-level embeddings: shape (768,) in output_folder_clip
      - Frame-level embeddings: shape (num_frames, 768) in output_folder_frame
    """
    model = get_model(ckpt_path).to(device)
    model.eval()

    in_dir = Path(parent_frames_dir)
    out_dir_clip = Path(output_folder_clip)
    out_dir_frame = Path(output_folder_frame)

    out_dir_clip.mkdir(parents=True, exist_ok=True)
    out_dir_frame.mkdir(parents=True, exist_ok=True)

    all_video_folders = sorted([d for d in in_dir.iterdir() if d.is_dir()])
    if not all_video_folders:
        print(f"⚠️ No subdirectories found in {parent_frames_dir}")
        return {"processed": 0}

    # Filter out folders where both clip and frame embeddings already exist
    if skip_existing:
        video_folders = [
            f
            for f in all_video_folders
            if not (
                (out_dir_clip / f"{f.name}.npy").exists()
                and (out_dir_frame / f"{f.name}.npy").exists()
            )
        ]
        skipped_count = len(all_video_folders) - len(video_folders)
        if skipped_count > 0:
            print(f"⏭️ Skipping {skipped_count} already extracted videos.")
    else:
        video_folders = all_video_folders

    if not video_folders:
        print("✅ All videos already have embeddings. Nothing to process.")
        return {"processed": 0}

    print(
        f"▶️ Processing {len(video_folders)} videos (Batch Size: {batch_size}, Frames/Vid: {num_frames}, Workers: {num_workers})..."
    )

    dataset = VideoFramesDataset(video_folders, num_frames=num_frames)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    saved_count = 0
    for batch_videos, batch_names in loader:
        # batch_videos shape: [B, num_frames, 3, 224, 224]
        B, n_f, C, H, W = batch_videos.shape

        # Flatten batch and frame dimensions for the ViT -> [B * num_frames, 3, 224, 224]
        batch_videos = batch_videos.view(B * n_f, C, H, W).to(
            device, non_blocking=True
        )

        with torch.no_grad():
            # 1. Batched forward pass -> [B * num_frames, 196, 768]
            video_tokens = forward_video_only(model, batch_videos)

            # 2. Reshape back to separate batch and frames -> [B, num_frames, 196, 768]
            video_tokens = video_tokens.view(B, n_f, 196, -1)

            # 3. Spatial pooling: Mean across 196 spatial tokens (dim=2) -> [B, num_frames, 768]
            frame_embs = video_tokens.mean(dim=2)

            # 4. Temporal pooling: Mean across num_frames (dim=1) -> [B, 768]
            video_embs = frame_embs.mean(dim=1)

            # 5. L2 normalization
            if l2_normalize:
                # Frame-level: normalize each of the 10 vectors along embedding dimension -> [B, num_frames, 768]
                frame_embs_norm = F.normalize(frame_embs, dim=-1)
                # Clip-level: normalize the temporally-pooled vector -> [B, 768]
                video_embs_norm = F.normalize(video_embs, dim=-1)
            else:
                frame_embs_norm = frame_embs
                video_embs_norm = video_embs

            # Move batch embeddings to CPU
            frame_embs_np = frame_embs_norm.cpu().numpy()
            video_embs_np = video_embs_norm.cpu().numpy()

        # 6. Save each pair individually as .npy
        for clip_emb, frame_mat, folder_name in zip(
            video_embs_np, frame_embs_np, batch_names
        ):
            np.save(str(out_dir_clip / f"{folder_name}.npy"), clip_emb)
            np.save(str(out_dir_frame / f"{folder_name}.npy"), frame_mat)
            saved_count += 1

        print(
            f" Saved {saved_count}/{len(video_folders)} videos...",
            end="\r",
            flush=True,
        )

    print(
        f"\n✅ Finished! Saved {saved_count} clip and frame embeddings to:\n   - {out_dir_clip}\n   - {out_dir_frame}"
    )
    return {"processed": saved_count}


# =====================================================================
# 4. FASTAPI LOCAL SERVER (Modern Lifespan Pattern)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before yield runs ON STARTUP
    get_model()
    yield
    # Code after yield runs ON SHUTDOWN (clean up memory if needed)


app = FastAPI(
    title="CAV-MAE Audio-Visual Service",
    description="Local service for audio feature extraction & cross-modal re-ranking",
    lifespan=lifespan,
)


class BatchAudioRequest(BaseModel):
    input_folder: str
    output_folder: str
    checkpoint_path: str = ""
    skip_existing: bool = True


@app.post("/extract_audio_batch")
def api_extract_audio(request: BatchAudioRequest):
    try:
        result = extract_audio_frames_batched(
            input_folder=request.input_folder,
            output_folder=request.output_folder,
            ckpt_path=request.checkpoint_path or None,
            skip_existing=request.skip_existing,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchVideoRequest(BaseModel):
    input_folder: str
    output_folder: str
    checkpoint_path: str = ""
    skip_existing: bool = True


@app.post("/extract_video_batch")
def api_extract_video(request: BatchVideoRequest):
    try:
        result = extract_video_embeddings_batched(
            parent_frames_dir=request.input_folder,
            output_folder=request.output_folder,
            ckpt_path=request.checkpoint_path or None,
            skip_existing=request.skip_existing,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

# =====================================================================
# 5. ENTRYPOINT & SERVER CONTROLLER
# =====================================================================
def start_server(port: int = 8000):
    """Call this function to launch the background FastAPI server."""
    print(f"🚀 Starting CAV-MAE Local Server on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    # Test extraction directly on local sample folder
    extract_audio_frames(
        input_folder="/home/hossein/Desktop/test_now", output_folder="/home/hossein/Desktop/test_now", ckpt_path=CAV_MAE_Scale_plusplus
    )
    extract_video_embeddings_from_folders(
        parent_frames_dir="/home/hossein/Desktop/test_now", output_folder="/home/hossein/Desktop/test_now", ckpt_path= CAV_MAE_Scale_plusplus, num_frames=5
    )


    def compute_similarity(audio_npy_path: str, video_npy_path: str):
        # Load and ensure 1D shape (768,)
        audio_emb = np.load(audio_npy_path).reshape(-1)
        video_emb = np.load(video_npy_path).reshape(-1)

        print(f"🎵 Audio Vector Shape: {audio_emb.shape}")
        print(f"🎬 Video Vector Shape: {video_emb.shape}")

        # Cosine Similarity: (A dot B) / (||A|| * ||B||)
        dot_product = np.dot(audio_emb, video_emb)
        norm_a = np.linalg.norm(audio_emb)
        norm_v = np.linalg.norm(video_emb)

        cos_sim = float(dot_product / (norm_a * norm_v + 1e-8))
        print(f"✨ Cosine Similarity:  {cos_sim:.4f}")
        return cos_sim
        
    base_pth = "/home/hossein/Desktop/test_now"
    print(compute_similarity(f"{base_pth}/-0gYWIOfqdM.npy", f"{base_pth}/-0gYWIOfqdM_GEN.npy"))
    print("***")
    print(compute_similarity(f"{base_pth}/-0gYWIOfqdM.npy", f"{base_pth}/-0gYWIOfqdM_OG.npy"))
    print("***")
    print(compute_similarity(f"{base_pth}/-0gYWIOfqdM.npy", f"{base_pth}/-0gYWIOfqdM_RAW_MIX.npy"))
    print("***")
    print(compute_similarity(f"{base_pth}/-jxXv_SoNaI.npy", f"{base_pth}/-0gYWIOfqdM.npy"))
    print("***")
    print(compute_similarity(f"{base_pth}/-0gYWIOfqdM_OG.npy", f"{base_pth}/-jxXv_SoNaI_OG.npy"))
    print("***")
    print(compute_similarity(f"{base_pth}/-jxXv_SoNaI.npy", f"{base_pth}/-jxXv_SoNaI_OG.npy"))
    # To run as server instead, uncomment:
    # start_server(port=8000)

    TRAIN_WAV_DIR = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/audios"
    TRAIN_FRAMES_DIR = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/frames"
    TRAIN_VIDEO_EMB_DIR = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/cav_mae/video_embeddings/10FrameMean"
    TRAIN_AUDIO_EMB_DIR = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/cav_mae/audio_embeddings"
    # extract_audio_frames(
    #     input_folder= TRAIN_WAV_DIR, output_folder=TRAIN_AUDIO_EMB_DIR, ckpt_path=CAV_MAE_Scale_plusplus
    # )
    # extract_video_embeddings_from_folders(
    #     parent_frames_dir=TRAIN_FRAMES_DIR, output_folder=TRAIN_VIDEO_EMB_DIR, ckpt_path= CAV_MAE_Scale_plusplus, num_frames=10
    # )
