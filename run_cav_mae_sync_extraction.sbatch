#!/bin/bash
#SBATCH --job-name="cav_mae_sync_extract"
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --exclude=tcml-node[1-2,5-13,16-17,20-36]
#SBATCH --cpus-per-task=6
#SBATCH --mem=45G
#SBATCH --time=06:00:00
#SBATCH --output=job_cav_mae_sync_%J.out
#SBATCH --error=job_cav_mae_sync_%J.err

set -e

PROJECT_DIR="/home/sherkat/CAV_MAE"
SCRATCH_DIR="/scratch/$SLURM_JOB_ID"

# Default flag values
INPUT_AUDIO_TAR=""
INPUT_FRAMES_TAR=""
OUTPUT_TAG="default_run"

# Parse CLI arguments passed to sbatch
# example use:
# sbatch extract.sbatch --audio_tar=/home/sherkat/vggsound_test_audio.tar --frames_tar=/home/sherkat/vggsound_test_frames.tar --output_tag=vggsound_test
while [[ $# -gt 0 ]]; do
  case $1 in
    --audio_tar=*)
      INPUT_AUDIO_TAR="${1#*=}"
      shift
      ;;
    --frames_tar=*)
      INPUT_FRAMES_TAR="${1#*=}"
      shift
      ;;
    --output_tag=*)
      OUTPUT_TAG="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

if [ -z "$INPUT_AUDIO_TAR" ] && [ -z "$INPUT_FRAMES_TAR" ]; then
    echo "❌ Error: You must specify at least --audio_tar=... or --frames_tar=..."
    exit 1
fi

# 1. Prepare target destinations using the output_tag
PROJECT_OUTPUT_DIR="$PROJECT_DIR/outputs_sync"
FINAL_DEST_AUDIO="$PROJECT_OUTPUT_DIR/$OUTPUT_TAG/audio_frame_embeddings"
FINAL_DEST_VIDEO="$PROJECT_OUTPUT_DIR/$OUTPUT_TAG/video_frame_embeddings"


# 2. Prepare scratch paths
export scratch_input_wavs="$SCRATCH_DIR/data/audio"
export scratch_input_frames="$SCRATCH_DIR/data/frames"
export scratch_audio_output="$SCRATCH_DIR/output_audio_frame_emb"
export scratch_video_output="$SCRATCH_DIR/output_vid_frame_emb"
mkdir -p "$scratch_input_wavs"
mkdir -p "$scratch_input_frames"
mkdir -p "$scratch_audio_output"
mkdir -p "$scratch_video_output"

# 3. Unpack input archives to scratch if provided
if [ -n "$INPUT_AUDIO_TAR" ] && [ -f "$INPUT_AUDIO_TAR" ]; then
    mkdir -p "$scratch_input_wavs"
    echo "📦 Unpacking audio from $INPUT_AUDIO_TAR to $scratch_input_wavs..."
    tar -xf "$INPUT_AUDIO_TAR" -C "$scratch_input_wavs" --strip-components=1
fi

if [ -n "$INPUT_FRAMES_TAR" ] && [ -f "$INPUT_FRAMES_TAR" ]; then
    mkdir -p "$scratch_input_frames"
    echo "📦 Unpacking frames from $INPUT_FRAMES_TAR to $scratch_input_frames..."
    tar -xf "$INPUT_FRAMES_TAR" -C "$scratch_input_frames" --strip-components=1
fi

cd "$PROJECT_DIR"

# 4. Run extraction
echo "🚀 Running CAV-MAE_SYNC extraction..."
uv run python tcml_cav_mae_sync.py

# 5. Sync results to permanent storage
if [ -n "$(ls -A "$scratch_audio_output" 2>/dev/null)" ]; then
    echo "💾 Copying audio embeddings to $FINAL_DEST_AUDIO..."
    mkdir -p "$FINAL_DEST_AUDIO"
    cp -r "$scratch_audio_output/"* "$FINAL_DEST_AUDIO/"
fi

if [ -n "$(ls -A "$scratch_video_output" 2>/dev/null)" ]; then
    echo "💾 Copying video frame embeddings to $FINAL_DEST_VIDEO..."
    mkdir -p "$FINAL_DEST_VIDEO"
    cp -r "$scratch_video_output/"* "$FINAL_DEST_VIDEO/"
fi

# 6. Scratch cleanup
echo "🧹 Cleaning up scratch directory..."
rm -rf "$SCRATCH_DIR"

echo "✅ Pipeline finished. Outputs saved under: $PROJECT_OUTPUT_DIR/$OUTPUT_TAG/"
