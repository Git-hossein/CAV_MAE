#!/bin/bash
#SBATCH --job-name="cav_mae_extract"
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --exclude=tcml-node[1-2,5-13,16-17,20-36]
#SBATCH --cpus-per-task=6
#SBATCH --mem=45G
#SBATCH --time=06:00:00
#SBATCH --output=job_cav_mae_%J.out
#SBATCH --error=job_cav_mae_%J.err

set -e

PROJECT_DIR="/home/sherkat/CAV_MAE"
SCRATCH_DIR="/scratch/$SLURM_JOB_ID"

# Default flag values
AUDIO_TAR=""
FRAMES_TAR=""
OUTPUT_TAG="default_run"

# Parse CLI arguments passed to sbatch
# example use:
# sbatch extract.sbatch --audio_tar=/home/sherkat/vggsound_test_audio.tar --frames_tar=/home/sherkat/vggsound_test_frames.tar --output_tag=vggsound_test
while [[ $# -gt 0 ]]; do
  case $1 in
    --audio_tar=*)
      AUDIO_TAR="${1#*=}"
      shift
      ;;
    --frames_tar=*)
      FRAMES_TAR="${1#*=}"
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

if [ -z "$AUDIO_TAR" ] && [ -z "$FRAMES_TAR" ]; then
    echo "❌ Error: You must specify at least --audio_tar=... or --frames_tar=..."
    exit 1
fi

# 1. Prepare target destinations using the output_tag
PROJECT_OUTPUT_DIR="$PROJECT_DIR/outputs"
DEST_AUDIO="$PROJECT_OUTPUT_DIR/$OUTPUT_TAG/audio_embeddings"
DEST_FRAME="$PROJECT_OUTPUT_DIR/$OUTPUT_TAG/video_frame_embeddings"
DEST_CLIP="$PROJECT_OUTPUT_DIR/$OUTPUT_TAG/video_clip_embeddings"

# 2. Prepare scratch paths
mkdir -p "$SCRATCH_DIR/output_audio_emb"
mkdir -p "$SCRATCH_DIR/output_vid_clip_emb"
mkdir -p "$SCRATCH_DIR/output_vid_frame_emb"

# 3. Unpack input archives to scratch if provided
if [ -n "$AUDIO_TAR" ] && [ -f "$AUDIO_TAR" ]; then
    mkdir -p "$SCRATCH_DIR/data/audio"
    echo "📦 Unpacking audio from $AUDIO_TAR to $SCRATCH_DIR/data/audio..."
    tar -xf "$AUDIO_TAR" -C "$SCRATCH_DIR/data/audio" --strip-components=1
fi

if [ -n "$FRAMES_TAR" ] && [ -f "$FRAMES_TAR" ]; then
    mkdir -p "$SCRATCH_DIR/data/frames"
    echo "📦 Unpacking frames from $FRAMES_TAR to $SCRATCH_DIR/data/frames..."
    tar -xf "$FRAMES_TAR" -C "$SCRATCH_DIR/data/frames" --strip-components=1
fi

cd "$PROJECT_DIR"

# 4. Run extraction
echo "🚀 Running CAV-MAE extraction..."
uv run python extract_embeddings.py

# 5. Sync results to permanent storage
if [ -n "$(ls -A "$SCRATCH_DIR/output_audio_emb" 2>/dev/null)" ]; then
    echo "💾 Copying audio embeddings to $DEST_AUDIO..."
    mkdir -p "$DEST_AUDIO"
    cp -r "$SCRATCH_DIR/output_audio_emb/"* "$DEST_AUDIO/"
fi

if [ -n "$(ls -A "$SCRATCH_DIR/output_vid_clip_emb" 2>/dev/null)" ]; then
    echo "💾 Copying video clip embeddings to $DEST_CLIP..."
    mkdir -p "$DEST_CLIP"
    cp -r "$SCRATCH_DIR/output_vid_clip_emb/"* "$DEST_CLIP/"
fi

if [ -n "$(ls -A "$SCRATCH_DIR/output_vid_frame_emb" 2>/dev/null)" ]; then
    echo "💾 Copying video frame embeddings to $DEST_FRAME..."
    mkdir -p "$DEST_FRAME"
    cp -r "$SCRATCH_DIR/output_vid_frame_emb/"* "$DEST_FRAME/"
fi

# 6. Scratch cleanup
echo "🧹 Cleaning up scratch directory..."
rm -rf "$SCRATCH_DIR"

echo "✅ Pipeline finished. Outputs saved under: $PROJECT_OUTPUT_DIR/$OUTPUT_TAG/"
