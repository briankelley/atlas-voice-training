#!/bin/bash
# Rebuild the training data tarball with all dependencies bundled
#
# This is a one-time migration script. It:
#   1. Downloads the 4 embedding models from dscripka's GitHub releases
#   2. Downloads MIT RIRs from MIT and converts to 16kHz mono
#   3. Downloads the existing tarball from HuggingFace
#   4. Extracts, adds the new content, repacks
#   5. Uploads the rebuilt tarball as a REPLACEMENT to HuggingFace
#   6. Uploads 5 individual model files for Dockerfile use
#
# Requirements: ~60GB free disk space, ffmpeg, python3 with huggingface_hub
#
# Usage: ./rebuild-tarball.sh [--staging-dir /path/to/staging]

set -e

HF_DATASET="brianckelley/atlas-voice-training-data"
HF_BASE="https://huggingface.co/datasets/${HF_DATASET}/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments
STAGING_DIR="${SCRIPT_DIR}/tarball-staging"
for arg in "$@"; do
    case "$arg" in
        --staging-dir) shift; STAGING_DIR="$1" ;;
    esac
done

# =============================================================================
# Preflight checks
# =============================================================================
echo "=============================================="
echo "Training Data Tarball Rebuild"
echo "=============================================="
echo ""

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is required for MIT RIR conversion."
    echo "  Install: sudo apt install ffmpeg"
    exit 1
fi

# Check HuggingFace auth
echo "[Preflight] Checking HuggingFace authentication..."
HF_USER=$(python3 -c "
from huggingface_hub import HfApi
try:
    info = HfApi().whoami()
    print(info['name'])
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)

if [[ "$HF_USER" == ERROR* ]]; then
    echo "  $HF_USER"
    echo "  Run: huggingface-cli login"
    exit 1
fi
echo "  Authenticated as: $HF_USER"

# Check disk space (use parent dir if staging doesn't exist yet)
DISK_CHECK_DIR="$STAGING_DIR"
[ ! -d "$DISK_CHECK_DIR" ] && DISK_CHECK_DIR="$(dirname "$STAGING_DIR")"
AVAIL_KB=$(df "$DISK_CHECK_DIR" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "  Staging directory: $STAGING_DIR"
echo "  Available disk: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 55 ]; then
    echo "  WARNING: Less than 55GB available. You need ~60GB for the full rebuild."
    read -p "  Continue anyway? [y/N]: " -r
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi
echo ""

mkdir -p "$STAGING_DIR"
cd "$STAGING_DIR"

# =============================================================================
# Step 1: Download embedding models from dscripka's GitHub releases
# =============================================================================
echo "[Step 1/6] Downloading embedding models from upstream releases..."
mkdir -p embedding_models

EMBEDDING_BASE="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
for model in embedding_model.onnx embedding_model.tflite melspectrogram.onnx melspectrogram.tflite; do
    if [ -f "embedding_models/$model" ]; then
        echo "  Already have: $model"
    else
        echo "  Downloading: $model"
        wget -q --show-progress -O "embedding_models/$model" "${EMBEDDING_BASE}/$model"
    fi
done
echo "  Embedding models:"
ls -lh embedding_models/
echo ""

# =============================================================================
# Step 2: Download and convert MIT Room Impulse Responses
# =============================================================================
echo "[Step 2/6] Downloading MIT Room Impulse Responses..."

if [ -d "mit_rirs" ] && [ -n "$(ls -A mit_rirs/*.wav 2>/dev/null)" ]; then
    MIT_COUNT=$(ls mit_rirs/*.wav | wc -l)
    echo "  Already have $MIT_COUNT WAV files in mit_rirs/. Skipping."
else
    MIT_ZIP="$STAGING_DIR/mit_rirs_download.zip"
    MIT_TEMP="$STAGING_DIR/mit_rirs_raw"
    mkdir -p mit_rirs "$MIT_TEMP"

    if [ ! -f "$MIT_ZIP" ]; then
        echo "  Downloading from MIT (~300MB)..."
        wget --progress=bar:force:noscroll -O "$MIT_ZIP" \
            "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
    else
        echo "  ZIP already downloaded."
    fi

    echo "  Extracting..."
    unzip -q -o "$MIT_ZIP" -d "$MIT_TEMP"

    echo "  Converting to 16kHz mono..."
    WAV_COUNT=0
    for f in $(find "$MIT_TEMP" -name "*.wav" -type f); do
        BASENAME=$(basename "$f")
        ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 "mit_rirs/$BASENAME"
        WAV_COUNT=$((WAV_COUNT + 1))
    done

    rm -rf "$MIT_ZIP" "$MIT_TEMP"
    echo "  Converted $WAV_COUNT room impulse responses to 16kHz."
fi
echo ""

# =============================================================================
# Step 3: Download existing tarball from HuggingFace
# =============================================================================
echo "[Step 3/6] Downloading existing tarball from HuggingFace (~20GB)..."

TARBALL="atlas-voice-training-data.tar.gz"
if [ -f "$TARBALL" ]; then
    echo "  Tarball already downloaded. Skipping."
else
    wget --progress=bar:force:noscroll -O "$TARBALL" \
        "${HF_BASE}/archive/${TARBALL}"
fi
echo ""

# =============================================================================
# Step 4: Extract existing tarball and add new content
# =============================================================================
echo "[Step 4/6] Extracting and rebuilding tarball..."

REBUILD_DIR="$STAGING_DIR/rebuild"
mkdir -p "$REBUILD_DIR"

echo "  Extracting existing tarball..."
tar -xzf "$TARBALL" -C "$REBUILD_DIR"

echo "  Adding MIT RIRs..."
cp -r mit_rirs "$REBUILD_DIR/mit_rirs"

echo "  Adding embedding models..."
cp -r embedding_models "$REBUILD_DIR/embedding_models"

echo ""
echo "  Rebuilt tarball contents:"
echo "  -------------------------"
for item in "$REBUILD_DIR"/*; do
    NAME=$(basename "$item")
    if [ -d "$item" ]; then
        COUNT=$(find "$item" -type f | wc -l)
        SIZE=$(du -sh "$item" | cut -f1)
        echo "    $NAME/  ($COUNT files, $SIZE)"
    else
        SIZE=$(du -h "$item" | cut -f1)
        echo "    $NAME  ($SIZE)"
    fi
done
echo "  -------------------------"
echo ""

read -p "  Contents look correct? [Y/n]: " -r
if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    echo "  Aborting. Staging directory preserved at: $STAGING_DIR"
    exit 0
fi

# =============================================================================
# Step 5: Repack tarball
# =============================================================================
echo "[Step 5/6] Repacking tarball..."

NEW_TARBALL="$STAGING_DIR/${TARBALL}.new"
cd "$REBUILD_DIR"
tar -I pigz -cf "$NEW_TARBALL" \
    openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
    validation_set_features.npy \
    musan_music/ \
    piper_tts_model/ \
    mit_rirs/ \
    embedding_models/
cd "$STAGING_DIR"

NEW_SIZE=$(du -h "$NEW_TARBALL" | cut -f1)
echo "  New tarball: $NEW_SIZE"
echo ""

# =============================================================================
# Step 6: Upload to HuggingFace
# =============================================================================
echo "[Step 6/6] Uploading to HuggingFace..."
echo ""
echo "  This will:"
echo "    1. Replace the existing tarball on HuggingFace"
echo "    2. Upload 4 embedding model files (individual)"
echo "    3. Restore the TTS model file (individual)"
echo ""
echo "  Dataset: $HF_DATASET"
echo "  Tarball size: $NEW_SIZE"
echo ""
read -p "  Proceed with upload? [Y/n]: " -r
if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    echo "  Upload skipped. Rebuilt tarball at: $NEW_TARBALL"
    exit 0
fi

echo ""
echo "  [6a] Uploading tarball (~20GB, this will take a while)..."
python3 << EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="${NEW_TARBALL}",
    path_in_repo="archive/atlas-voice-training-data.tar.gz",
    repo_id="${HF_DATASET}",
    repo_type="dataset",
)
print("  Tarball uploaded.")
EOF

echo ""
echo "  [6b] Uploading embedding models (individual files)..."
for model in embedding_model.onnx embedding_model.tflite melspectrogram.onnx melspectrogram.tflite; do
    echo "    Uploading: embedding_models/$model"
    python3 << EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="${STAGING_DIR}/embedding_models/${model}",
    path_in_repo="embedding_models/${model}",
    repo_id="${HF_DATASET}",
    repo_type="dataset",
)
EOF
done
echo "  Embedding models uploaded."

echo ""
echo "  [6c] Restoring TTS model (individual file)..."
python3 << EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="${REBUILD_DIR}/piper_tts_model/en-us-libritts-high.pt",
    path_in_repo="piper_tts_model/en-us-libritts-high.pt",
    repo_id="${HF_DATASET}",
    repo_type="dataset",
)
print("  TTS model uploaded.")
EOF

echo ""
echo "=============================================="
echo "  REBUILD COMPLETE"
echo "=============================================="
echo ""
echo "  Tarball replaced on HuggingFace"
echo "  Individual model files uploaded"
echo ""
echo "  Verify with:"
echo "    curl -sI '${HF_BASE}/archive/atlas-voice-training-data.tar.gz' | head -3"
echo "    curl -sI '${HF_BASE}/embedding_models/embedding_model.onnx' | head -3"
echo "    curl -sI '${HF_BASE}/piper_tts_model/en-us-libritts-high.pt' | head -3"
echo ""
echo "  To clean up staging (~60GB):"
echo "    rm -rf $STAGING_DIR"
echo ""
