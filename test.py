import sys
import torch

# Add the cloned cav_mae directory to Python's module search path
sys.path.append("./cav_mae")

from src.models.cav_mae import CAVMAE

# 1. Set execution device to CPU
device = torch.device("cpu")
print(f"Running on: {device}")

# 2. Initialize CAV-MAE with standard AudioSet-pretrained settings
model = CAVMAE(
    audio_length=1024,  # Standard 10-second audio spectrogram length
    modality_specific_depth=11,  # 11 unimodal transformer layers + 1 shared layer
    norm_pix_loss=True,  # Normalized pixel loss flag used in pretraining
    tr_pos=False,  # Non-trainable positional embeddings
)

# 3. Move model to device and set to evaluation mode
model.to(device)
model.eval()

print("✅ CAV-MAE initialized successfully on CPU!")