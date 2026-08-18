from pathlib import Path
from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import torch
from PIL import Image
import torchvision.transforms as T
import torchaudio
import uvicorn
import soundfile as sf

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
            norm_pix_loss=True,
            tr_pos=False,
        )
        if ckpt_path and Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = (
                checkpoint["model"] if "model" in checkpoint else checkpoint
            )
            _model_instance.load_state_dict(state_dict, strict=False)
            print(f" Loaded checkpoint weights from: {ckpt_path}")
        else:
            print("⚠️ Running with randomly initialized weights (No checkpoint).")

        _model_instance.to(device)
        _model_instance.eval()
        print(" CAV-MAE model loaded and ready.")
    return _model_instance



# =====================================================================
# 2. AUDIO BATCH EXTRACTION FUNCTION
# =====================================================================


def preprocess_audio(
    wav_path: str,
    target_sample_rate: int = 16000,
    target_length: int = 1024,
    norm_mean: float = -4.268,
    norm_std: float = 4.569,
) -> torch.Tensor:
    """Preprocesses raw .wav audio into the (1, 1024, 128) tensor CAV-MAE expects."""
    # 1. Load Audio using soundfile (bypasses torchcodec)
    data, sr = sf.read(wav_path, dtype="float32")
    waveform = torch.from_numpy(data)

    # Handle shape: sf.read returns (samples,) or (samples, channels)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, samples]
    else:
        waveform = waveform.t()  # [channels, samples]

    # 2. Convert multi-channel to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 3. Resample to 16kHz if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # 4. Compute 128 Mel Filterbanks
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=target_sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )  # Output shape: [time_frames, 128]

    # 5. Pad or truncate to target_length (1024 frames = ~10 seconds)
    n_frames = fbank.shape[0]
    if n_frames < target_length:
        pad = torch.zeros((target_length - n_frames, 128))
        fbank = torch.cat([fbank, pad], dim=0)
    elif n_frames > target_length:
        fbank = fbank[:target_length, :]

    # 6. Normalize with AudioSet dataset stats
    fbank = (fbank - norm_mean) / (norm_std * 2)

    # Return shape: [1, 1024, 128]
    return fbank.unsqueeze(0).to(device)


def extract_audio_frames(
    input_folder: str, output_folder: str, ckpt_path: str = None, skip_existing = True
):
    """Iterates over all .wav files in input_folder, extracts CAV-MAE

    embeddings (single pooled vector), and saves matching .npy files.
    """
    model = get_model(ckpt_path)
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(in_dir.glob("*.wav")))
    if not wav_files:
        print(f"⚠️ No .wav files found in {input_folder}")
        return {"processed": 0, "saved_to": str(out_dir)}

    print(f"▶️ Found {len(wav_files)} audio files. Starting extraction...")

    saved_count = 0
    for wav_file in wav_files:
        out_file = out_dir / f"{wav_file.stem}.npy"
        
        if skip_existing and out_file.exists():
            print(f"⏭️ Embedding for {wav_file.name} already exists. Skipping...")
            continue

        try:
            # 1. Preprocess audio tensor
            audio_tensor = preprocess_audio(str(wav_file))

            # 2. Extract unimodal audio features with CAV-MAE
            with torch.no_grad():
                # 1. Add channel dimension: CAV-MAE audio patch embedder expects [B, 1, 1024, 128]
                if audio_tensor.ndim == 3:
                    a_input = audio_tensor.unsqueeze(1)
                else:
                    a_input = audio_tensor

                # 2. Patch embedding + positional embedding
                a_tokens = model.patch_embed_a(a_input)
                a_tokens = a_tokens + model.pos_embed_a

                # 3. Pass through the audio Transformer blocks (modality-specific depth = 11)
                for blk in model.blocks_a:
                    a_tokens = blk(a_tokens)

                # LayerNorm before pooling (if present)
                if hasattr(model, "norm_a"):
                    a_tokens = model.norm_a(a_tokens)

                # 4. Mean pool over the 512 patch tokens -> (768,)
                audio_emb = a_tokens.mean(dim=1).squeeze(0).cpu().numpy()

            # 3. Save as .npy
            np.save(str(out_file), audio_emb)
            saved_count += 1

        except Exception as e:
            print(f"❌ Failed processing {wav_file.name}: {e}")

    print(f" Finished! Saved {saved_count}/{len(wav_files)} embeddings to: {out_dir}")
    return {"processed": saved_count, "saved_to": str(out_dir)}



# =====================================================================
# 3. VIDEO BATCH EXTRACTION FUNCTION
# =====================================================================


video_transform = T.Compose(
    [
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_video_from_frame_folder(
    frame_dir: Path, num_frames: int = 16
) -> torch.Tensor:
    frame_paths = sorted(list(frame_dir.glob("*.jpg")))
    total_frames = len(frame_paths)

    if total_frames == 0:
        raise ValueError(f"No .jpg frames found in: {frame_dir}")

    # Uniformly sample 16 indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        with Image.open(frame_paths[idx]) as img:
            frames.append(video_transform(img.convert("RGB")))

    # Stack to [16, 3, 224, 224] -> (16 frames, 3 RGB channels)
    video_tensor = torch.stack(frames, dim=0)
    return video_tensor.to(device)


def extract_video_embeddings_from_folders(
    parent_frames_dir: str, output_folder: str, ckpt_path: str = None, skip_existing = True
):
    """Iterates over subdirectories (one per video) in parent_frames_dir,

    extracts CAV-MAE 1D pooled embeddings, and saves matching .npy files.
    """
    model = get_model(ckpt_path)
    in_dir = Path(parent_frames_dir)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all subdirectories that contain frames
    video_folders = sorted([d for d in in_dir.iterdir() if d.is_dir()])

    if not video_folders:
        print(f"⚠️ No subdirectories found in {parent_frames_dir}")
        return {"processed": 0, "saved_to": str(out_dir)}

    print(
        f"▶️ Found {len(video_folders)} video frame folders. Starting extraction..."
    )

    saved_count = 0
    for folder in video_folders:
        out_file = out_dir / f"{folder.name}.npy"

        if skip_existing and out_file.exists():
            print(f"⏭️ Embedding for {folder.name} already exists. Skipping...")
            continue

        try:
            # 1. video_tensor shape: [16, 3, 224, 224]
            video_tensor = load_video_from_frame_folder(folder)

            # 2. Extract visual features with CAV-MAE
            with torch.no_grad():
                # Pass 16 frames through 2D patch embedder -> [16, 196, 768]
                v_tokens = model.patch_embed_v(video_tensor)

                # Flatten across frames and spatial patches -> [1, 16 * 196, 768] (or mean across frames)
                # Reshape to [1, 16 * 196, 768] to add pos_embed_v if pos_embed_v has matching token length,
                # or process per-frame and mean-pool:
                if (
                    hasattr(model, "pos_embed_v")
                    and model.pos_embed_v.shape[1] == v_tokens.shape[1]
                ):
                    # pos_embed_v is per-frame [1, 196, 768]
                    v_tokens = v_tokens + model.pos_embed_v
                    for blk in model.blocks_v:
                        v_tokens = blk(v_tokens)

                    # Apply visual LayerNorm
                    if hasattr(model, "norm_v"):
                        v_tokens = model.norm_v(v_tokens)
                        
                    # Pool across all patches and frames -> (768,)
                    video_emb = v_tokens.mean(dim=(0, 1)).cpu().numpy()
                else:
                    # Flattened token representation [1, N, 768]
                    v_tokens = v_tokens.view(1, -1, v_tokens.shape[-1])
                    if hasattr(model, "pos_embed_v"):
                        v_tokens = v_tokens + model.pos_embed_v
                    for blk in model.blocks_v:
                        v_tokens = blk(v_tokens)

                    # Apply visual LayerNorm
                    if hasattr(model, "norm_v"):
                        v_tokens = model.norm_v(v_tokens)

                    video_emb = v_tokens.mean(dim=1).squeeze(0).cpu().numpy()

            # 3. Save matching .npy
            np.save(str(out_file), video_emb)
            saved_count += 1

        except Exception as e:
            print(f"❌ Failed processing {folder.name}: {e}")

    print(
        f" Finished! Saved {saved_count}/{len(video_folders)} video embeddings to: {out_dir}"
    )
    return {"processed": saved_count, "saved_to": str(out_dir)}


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
        result = extract_audio_frames(
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
        result = extract_video_embeddings_from_folders(
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
        parent_frames_dir="/home/hossein/Desktop/test_now", output_folder="/home/hossein/Desktop/test_now", ckpt_path= CAV_MAE_Scale_plusplus
    )


    def compute_similarity(audio_npy_path: str, video_npy_path: str):
        # 1. Load the 768-d vectors
        audio_emb = np.load(audio_npy_path)
        video_emb = np.load(video_npy_path)
        
        # 2. Compute Cosine Similarity: (A dot B) / (||A|| * ||B||)
        dot_product = np.dot(audio_emb, video_emb)
        norm_a = np.linalg.norm(audio_emb)
        norm_v = np.linalg.norm(video_emb)
        
        cos_sim = dot_product / (norm_a * norm_v)
        
        print(f"🎵 Audio Vector Shape: {audio_emb.shape}")
        print(f"🎬 Video Vector Shape: {video_emb.shape}")
        print(f"✨ Cosine Similarity:  {cos_sim:.4f}")
        
        return cos_sim
    
    base_pth = "/home/hossein/Desktop/test_now"
    print(compute_similarity(f"{base_pth}/-0gYWIOfqdM.npy", f"{base_pth}/-0gYWIOfqdM_GEN.npy"))
    # To run as server instead, uncomment:
    # start_server(port=8000)