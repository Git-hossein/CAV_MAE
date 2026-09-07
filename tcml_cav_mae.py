from pathlib import Path
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

# Path to CAV-MAE model repository
sys.path.append("./cav_mae")
from src.models.cav_mae import CAVMAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_instance = None

# Default output locations on local scratch
DEFAULT_OUTPUT_AUDIO_EMB = os.path.expandvars("/scratch/$SLURM_JOB_ID/output_audio_emb")
DEFAULT_OUTPUT_VID_CLIP_EMB = os.path.expandvars("/scratch/$SLURM_JOB_ID/output_vid_clip_emb")
DEFAULT_OUTPUT_VID_FRAME_EMB = os.path.expandvars("/scratch/$SLURM_JOB_ID/output_vid_frame_emb")


def get_model(ckpt_path: str = None):
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
            clean_state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            msg = _model_instance.load_state_dict(clean_state_dict, strict=False)
            print(f" Loaded checkpoint weights from: {ckpt_path}")
            print(f"   Missing/Unexpected keys: {msg}")
        else:
            print("⚠️ Running with randomly initialized weights (No checkpoint).")

        _model_instance.to(device)
        _model_instance.eval()
        print(" CAV-MAE model loaded and ready.")
    return _model_instance


def forward_audio_only(model: torch.nn.Module, a: torch.Tensor) -> torch.Tensor:
    a = a.unsqueeze(1)
    a = a.transpose(2, 3)
    a = model.patch_embed_a(a)
    a = a + model.pos_embed_a
    a = a + model.modality_a

    for blk in model.blocks_a:
        a = blk(a)
    for blk in model.blocks_u:
        a = blk(a, "a")

    return model.norm_a(a)


def forward_video_only(model: torch.nn.Module, v: torch.Tensor) -> torch.Tensor:
    v = model.patch_embed_v(v)
    v = v + model.pos_embed_v
    v = v + model.modality_v

    for blk in model.blocks_v:
        v = blk(v)
    for blk in model.blocks_u:
        v = blk(v, "v")

    return model.norm_v(v)


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
        data, sr = sf.read(str(wav_path), dtype="float32")
        waveform = torch.from_numpy(data)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.target_sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            pad = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = pad(fbank)
        elif p < 0:
            fbank = fbank[: self.target_length, :]

        fbank = (fbank - self.norm_mean) / self.norm_std
        return fbank, str(wav_path)


def extract_audio_frames_batched(
    input_folder: str,
    output_folder: str,
    ckpt_path: str = "audio_model.21.pth",
    batch_size: int = 32,
    num_workers: int = 4,
    skip_existing: bool = True,
    l2_normalize: bool = True,
):
    model = get_model(ckpt_path).to(device)
    model.eval()

    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_wav_files = sorted(list(in_dir.glob("*.wav")))
    if not all_wav_files:
        print(f"⚠️ No .wav files found in {input_folder}")
        return

    if skip_existing:
        wav_files = [p for p in all_wav_files if not (out_dir / f"{p.stem}.npy").exists()]
        skipped_count = len(all_wav_files) - len(wav_files)
        if skipped_count > 0:
            print(f"⏭️ Skipping {skipped_count} already extracted audio files.")
    else:
        wav_files = all_wav_files

    if not wav_files:
        print("✅ All audio files already have embeddings.")
        return

    print(f"▶️ Processing {len(wav_files)} audio files (Batch Size: {batch_size}, Workers: {num_workers})...")
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
        batch_fbanks = batch_fbanks.to(device, non_blocking=True)

        with torch.no_grad():
            audio_tokens = forward_audio_only(model, batch_fbanks)
            audio_embs = audio_tokens.mean(dim=1)
            if l2_normalize:
                audio_embs = F.normalize(audio_embs, dim=-1)
            audio_embs = audio_embs.cpu().numpy()

        for emb, path_str in zip(audio_embs, batch_paths):
            stem = Path(path_str).stem
            out_file = out_dir / f"{stem}.npy"
            np.save(str(out_file), emb)
            saved_count += 1

        print(f" Saved {saved_count}/{len(wav_files)} audio embeddings...", end="\r", flush=True)

    print(f"\n✅ Finished! Saved {saved_count} audio embeddings to: {output_folder}")


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
        frame_paths = sorted(
            list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
        )
        total_frames = len(frame_paths)
        if total_frames == 0:
            raise ValueError(f"No frames found in: {folder_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for i in indices:
            with Image.open(frame_paths[i]) as img:
                frames.append(video_transform(img.convert("RGB")))

        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, folder_path.name


def extract_video_embeddings_batched(
    parent_frames_dir: str,
    output_folder_clip: str,
    output_folder_frame: str,
    ckpt_path: str = "audio_model.21.pth",
    num_frames: int = 10,
    batch_size: int = 8,
    num_workers: int = 4,
    skip_existing: bool = True,
    l2_normalize: bool = True,
):
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
        return

    if skip_existing:
        video_folders = [
            f for f in all_video_folders
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
        print("✅ All videos already have embeddings.")
        return

    print(f"▶️ Processing {len(video_folders)} videos (Batch Size: {batch_size}, Frames/Vid: {num_frames}, Workers: {num_workers})...")
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
        B, n_f, C, H, W = batch_videos.shape
        batch_videos = batch_videos.view(B * n_f, C, H, W).to(device, non_blocking=True)

        with torch.no_grad():
            video_tokens = forward_video_only(model, batch_videos)
            video_tokens = video_tokens.view(B, n_f, 196, -1)
            frame_embs = video_tokens.mean(dim=2)
            video_embs = frame_embs.mean(dim=1)

            if l2_normalize:
                frame_embs_norm = F.normalize(frame_embs, dim=-1)
                video_embs_norm = F.normalize(video_embs, dim=-1)
            else:
                frame_embs_norm = frame_embs
                video_embs_norm = video_embs

            frame_embs_np = frame_embs_norm.cpu().numpy()
            video_embs_np = video_embs_norm.cpu().numpy()

        for clip_emb, frame_mat, folder_name in zip(video_embs_np, frame_embs_np, batch_names):
            np.save(str(out_dir_clip / f"{folder_name}.npy"), clip_emb)
            np.save(str(out_dir_frame / f"{folder_name}.npy"), frame_mat)
            saved_count += 1

        print(f" Saved {saved_count}/{len(video_folders)} videos...", end="\r", flush=True)

    print(f"\n✅ Finished! Saved {saved_count} clip and frame embeddings to:\n   - {out_dir_clip}\n   - {out_dir_frame}")


if __name__ == "__main__":
    scratch_job_id = os.environ.get("SLURM_JOB_ID", "")
    base_scratch = Path(f"/scratch/{scratch_job_id}") if scratch_job_id else Path("./scratch")

    INPUT_SCRATCH_AUDIOS = base_scratch / "data" / "audio"
    INPUT_SCRATCH_FRAMES = base_scratch / "data" / "frames"

    OUTPUT_AUDIO_EMB = base_scratch / "output_audio_emb"
    OUTPUT_VID_CLIP_EMB = base_scratch / "output_vid_clip_emb"
    OUTPUT_VID_FRAME_EMB = base_scratch / "output_vid_frame_emb"

    ran_any = False

    # Run Audio Extraction if audio scratch folder exists and has files
    if INPUT_SCRATCH_AUDIOS.exists() and any(INPUT_SCRATCH_AUDIOS.iterdir()):
        print(f"🎵 Found audio data in {INPUT_SCRATCH_AUDIOS}, starting audio extraction...")
        extract_audio_frames_batched(
            input_folder=str(INPUT_SCRATCH_AUDIOS),
            output_folder=str(OUTPUT_AUDIO_EMB),
            ckpt_path="./audio_model.21.pth",
            batch_size=64,
            num_workers=6,
        )
        ran_any = True

    # Run Video Extraction if frames scratch folder exists and has files
    if INPUT_SCRATCH_FRAMES.exists() and any(INPUT_SCRATCH_FRAMES.iterdir()):
        print(f"🎬 Found frame data in {INPUT_SCRATCH_FRAMES}, starting video extraction...")
        extract_video_embeddings_batched(
            parent_frames_dir=str(INPUT_SCRATCH_FRAMES),
            output_folder_clip=str(OUTPUT_VID_CLIP_EMB),
            output_folder_frame=str(OUTPUT_VID_FRAME_EMB),
            ckpt_path="./audio_model.21.pth",
            num_frames=10,
            batch_size=8,
            num_workers=6,
        )
        ran_any = True

    if not ran_any:
        print("⚠️ No audio or frame data found in scratch. Nothing processed.")