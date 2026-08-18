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

    # Pad or truncate
    n_frames = fbank.shape[0]
    if n_frames < target_length:
        pad = torch.zeros((target_length - n_frames, 128))
        fbank = torch.cat([fbank, pad], dim=0)
    elif n_frames > target_length:
        fbank = fbank[:target_length, :]

    # Normalize with CAV-MAE AudioSet statistics
    fbank = (fbank - norm_mean) / (norm_std * 2)

    # Shape: [1, 1024, 128]
    return fbank.unsqueeze(0)


def extract_audio_frames(
    input_folder: str, output_folder: str, ckpt_path: str = None, skip_existing: bool = True
):
    """Iterates over all .wav files in input_folder, extracts CAV-MAE pooled embeddings,

    and saves matching .npy files.
    """
    model = get_model(ckpt_path).to(device)
    model.eval()

    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(in_dir.glob("*.wav")))
    if not wav_files:
        print(f"⚠️ No .wav files found in {input_folder}")
        return {"processed": 0, "saved_to": str(out_dir)}

    print(f"▶️ Found {len(wav_files)} audio files. Starting extraction...")

    # Dummy video tensor for unimodal audio inference [B, Frames, C, H, W]
    dummy_video = torch.zeros(5, 3, 224, 224, device=device)

    saved_count = 0
    for wav_file in wav_files:
        out_file = out_dir / f"{wav_file.stem}.npy"
        
        if skip_existing and out_file.exists():
            print(f"⏭️ Embedding for {wav_file.name} already exists. Skipping...")
            continue

        try:
            # 1. Preprocess audio: [1, 1024, 128]
            audio_tensor = preprocess_audio(str(wav_file)).to(device)

            # 2. Extract features via forward_feat
            with torch.no_grad():
                # forward_feat outputs: (audio_feat [B, 512, 768], video_feat)
                audio_tokens, _ = model.forward_feat(audio_tensor, dummy_video)
                
                # Mean-pool over 512 patch tokens -> [768]
                audio_emb = audio_tokens.mean(dim=1).squeeze(0).cpu().numpy()

            # 3. Save as .npy
            np.save(str(out_file), audio_emb)
            saved_count += 1

        except Exception as e:
            print(f"❌ Failed processing {wav_file.name}: {e}")

    print(f"✅ Finished! Saved {saved_count}/{len(wav_files)} embeddings to: {out_dir}")
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
    frame_dir: Path, num_frames: int = 5
) -> torch.Tensor:
    frame_paths = sorted(list(frame_dir.glob("*.jpg")))
    total_frames = len(frame_paths)

    if total_frames == 0:
        raise ValueError(f"No .jpg frames found in: {frame_dir}")

    # CAV-MAE expects exactly 5 frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        with Image.open(frame_paths[idx]) as img:
            frames.append(video_transform(img.convert("RGB")))

    # Output shape: [5, 3, 224, 224] (No unsqueeze!)
    video_tensor = torch.stack(frames, dim=0)
    return video_tensor.to(device)

def extract_video_embeddings_from_folders(
    parent_frames_dir: str, output_folder: str, ckpt_path: str = None, skip_existing: bool = True
):
    model = get_model(ckpt_path).to(device)
    model.eval()

    in_dir = Path(parent_frames_dir)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_folders = sorted([d for d in in_dir.iterdir() if d.is_dir()])
    if not video_folders:
        print(f"⚠️ No subdirectories found in {parent_frames_dir}")
        return {"processed": 0, "saved_to": str(out_dir)}

    # Match dummy audio batch dimension to the number of frames (5)
    dummy_audio = torch.zeros(5, 1024, 128, device=device)

    saved_count = 0
    for folder in video_folders:
        out_file = out_dir / f"{folder.name}.npy"
        if skip_existing and out_file.exists():
            continue

        try:
            # video_tensor shape: [5, 3, 224, 224]
            video_tensor = load_video_from_frame_folder(folder, num_frames=5)

            with torch.no_grad():
                # _, video_tokens shape: [5, 196, 768]
                _, video_tokens = model.forward_feat(dummy_audio, video_tensor)

                # Mean-pool across all 196 spatial patches AND all 5 frames -> Shape: [768]
                video_emb = video_tokens.mean(dim=(0, 1)).cpu().numpy()

            np.save(str(out_file), video_emb)
            saved_count += 1

        except Exception as e:
            print(f"❌ Failed processing {folder.name}: {e}")

    print(f"✅ Finished! Saved {saved_count}/{len(video_folders)} video embeddings to: {out_dir}")
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