#!/bin/bash
# Wrapper script to build and run atlas-voice training in Docker
#
# Usage: ./train-wakeword.sh [--rebuild] [--standalone]
#
# Options:
#   --rebuild      Force rebuild of Docker image even if it exists
#   --standalone   Run without local training data (downloads ~20GB from remote)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="atlas-voice-training"
CONTAINER_NAME="atlas-training-$(date +%Y%m%d-%H%M%S)"

# Training data location (where the big .npy files are)
# Set this to your local training data path, or leave empty for standalone mode
DATA_DIR="${ATLAS_DATA_DIR:-}"

# Output directory for trained models
OUTPUT_DIR="$SCRIPT_DIR/docker-output"
mkdir -p "$OUTPUT_DIR"

# Tarball download URL for standalone mode
TARBALL_URL="${TARBALL_URL:-https://huggingface.co/datasets/brianckelley/atlas-voice-training-data/resolve/main/archive/atlas-voice-training-data.tar.gz}"

# Parse arguments
REBUILD=false
STANDALONE=false
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=true ;;
        --standalone) STANDALONE=true ;;
        --preflight) PREFLIGHT_ONLY=true ;;
    esac
done

# Auto-detect standalone mode if local data doesn't exist
if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    STANDALONE=true
fi

# =========================================================================
# Pre-flight checks
# =========================================================================

# Check for NVIDIA GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$GPU_NAME" ]; then
    echo "  GPU:                  FAILED -- nvidia-smi not found"
    echo ""
    echo "  GPU required for training."
    exit 1
else
    echo "  GPU:                  $GPU_NAME"
fi

# Check Docker is running
if ! docker info &>/dev/null; then
    echo "  Docker:               FAILED"
    echo ""
    if ! systemctl is-active --quiet docker 2>/dev/null; then
        echo "  Docker service is not running."
        echo "  Fix: sudo systemctl start docker"
    elif ! groups | grep -q docker; then
        echo "  Your user is not in the 'docker' group."
        echo "  Fix: sudo usermod -aG docker \$USER"
        echo "  Then log out and back in (or run: newgrp docker)"
    else
        echo "  Docker may not be installed."
        echo "  See: https://docs.docker.com/engine/install/"
    fi
    exit 1
else
    DOCKER_VER=$(docker --version 2>/dev/null | sed 's/Docker version //' | cut -d, -f1)
    echo "  Docker:               $DOCKER_VER"
fi

# Check NVIDIA container runtime
NVIDIA_PREFLIGHT_OK=true

if ! dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q ^ii; then
    echo "  NVIDIA runtime:       FAILED -- not installed"
    echo ""
    echo "  nvidia-container-toolkit lets Docker containers access the GPU."
    echo ""
    echo "  Install it:"
    echo "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "    echo \"deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(dpkg --print-architecture) /\" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
    echo ""
    NVIDIA_PREFLIGHT_OK=false
elif ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|Default Runtime.*nvidia"; then
    echo "  NVIDIA runtime:       FAILED -- not configured"
    echo ""
    echo "  nvidia-container-toolkit is installed but Docker is not configured to use it."
    echo ""
    echo "  Fix:"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
    echo ""
    NVIDIA_PREFLIGHT_OK=false
else
    echo "  NVIDIA runtime:       OK"
fi

# Check Docker image
if docker image inspect "$IMAGE_NAME" &>/dev/null; then
    IMG_CREATED=$(docker image inspect "$IMAGE_NAME" --format '{{.Created}}' 2>/dev/null | cut -dT -f1)
    echo "  Docker image:         built $IMG_CREATED"
else
    echo "  Docker image:         not built (will build on first run)"
fi

echo ""

if [ "$NVIDIA_PREFLIGHT_OK" = false ]; then
    read -p "  Try anyway? [y/N] " -r
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

if [ "${PREFLIGHT_ONLY:-false}" = true ]; then
    exit 0
fi

# Check for required files in local mode
if [ "$STANDALONE" == "false" ]; then
    for f in openwakeword_features_ACAV100M_2000_hrs_16bit.npy validation_set_features.npy; do
        if [ ! -f "$DATA_DIR/$f" ]; then
            echo "ERROR: Required file not found: $DATA_DIR/$f"
            exit 1
        fi
    done
fi

# =========================================================================
# Build Docker image
# =========================================================================
if ! docker image inspect "$IMAGE_NAME" &>/dev/null || [ "$REBUILD" == "true" ]; then
    echo "Building Docker image (this takes a few minutes the first time)..."
    echo ""
    cd "$SCRIPT_DIR"
    set +e
    docker build -f Dockerfile.training -t "$IMAGE_NAME" .
    BUILD_EXIT=$?
    set -e

    if [ $BUILD_EXIT -ne 0 ]; then
        echo ""
        echo "ERROR: Docker build failed."
        echo ""
        echo "  If the error mentions 'content store' or"
        echo "  'failed to get reader', run:"
        echo ""
        echo "    docker system prune -a"
        echo "    docker buildx prune -a"
        echo ""
        echo "  Then rerun this script."
        exit 1
    fi

    echo ""
    echo "Docker image built successfully."
    echo ""
else
    echo "Using existing Docker image: $IMAGE_NAME"
    echo ""
fi

# =========================================================================
# User interaction - proceed, wake word, training settings
# =========================================================================

echo "=============================================="
echo "  OpenWakeWord Custom Wake Word Training"
echo "=============================================="
echo ""
echo "The training environment is ready. Next we need"
echo "to download ~20 GB of training data to train a"
echo "custom wake word model."
echo ""
echo "  Source: $TARBALL_URL"
echo ""
echo "You can open that URL in a browser to inspect"
echo "the dataset before downloading."
echo ""

read -p "Do you want to proceed? [y/N] " -r
echo ""
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "To download training data and train a new word"
    echo "later, just rerun this script (./train-wakeword.sh)"
    exit 0
fi

# --- Wake word ---
echo "----------------------------------------------"
echo "  Wake Word Configuration"
echo "----------------------------------------------"
echo ""
echo "What wake word do you want to train?"
echo ""
echo "Recommended: Two-word phrases work best. \"Hey Atlas\""
echo "consistently outperformed \"Atlas\" by 10+ points in"
echo "accuracy and 18+ points in recall (not having to"
echo "repeat yourself) in tested configurations."
echo ""
echo "Examples: \"Hey Atlas\", \"Hey Jarvis\", \"Okay Computer\""
echo ""
read -p "Wake word: " WAKE_WORD
echo ""

if [ -z "$WAKE_WORD" ]; then
    echo "ERROR: Wake word cannot be empty."
    exit 1
fi

# Derive model name from wake word (lowercase, spaces to underscores)
MODEL_NAME=$(echo "$WAKE_WORD" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# --- Shared memory ---
# Docker defaults /dev/shm to 64MB which is too small for PyTorch DataLoader.
# Auto-detect a reasonable size based on available RAM.
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
AUTO_SHM=$((TOTAL_RAM_GB / 4))
# Floor at 2, cap at 32
[ "$AUTO_SHM" -lt 2 ] && AUTO_SHM=2
[ "$AUTO_SHM" -gt 32 ] && AUTO_SHM=32

echo "----------------------------------------------"
echo "  Docker Shared Memory"
echo "----------------------------------------------"
echo ""
echo "PyTorch's data loader needs more shared memory"
echo "than Docker's 64 MB default. This setting controls"
echo "how much of your system RAM the container is"
echo "allowed to use for shared memory."
echo ""
echo "  System RAM detected:  ${TOTAL_RAM_GB} GB"
echo "  Recommended:          ${AUTO_SHM} GB"
echo ""
read -p "Shared memory in GB [default: $AUTO_SHM]: " INPUT
SHM_SIZE="${INPUT:-$AUTO_SHM}g"
echo ""

# --- Training settings ---
# Defaults (Jan 30 model - best balance of accuracy and recall)
N_SAMPLES=50000
N_SAMPLES_VAL=5000
AUGMENTATION_ROUNDS=2
TRAINING_STEPS=100000
LAYER_SIZE=32

echo "----------------------------------------------"
echo "  Training Settings"
echo "----------------------------------------------"
echo ""
echo "These are the default settings that will be used"
echo "unless you decide to change them. They produce the"
echo "best balance of accuracy and recall."
echo ""
echo "  Training samples:     $N_SAMPLES"
echo "  Augmentation rounds:  $AUGMENTATION_ROUNDS"
echo "  Training steps:       $TRAINING_STEPS"
echo "  Layer size (neurons): $LAYER_SIZE"
echo ""
read -p "Use recommended settings? (select \"n\" for more info) [Y/n] " -r
echo ""

if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    # --- Training samples ---
    echo "----------------------------------------------"
    echo "  Training Samples              default: 50000"
    echo "----------------------------------------------"
    echo "How many synthetic speech clips to generate"
    echo "for training."
    echo ""
    echo "More samples = more pronunciation variety."
    echo "Fewer samples = faster training."
    echo ""
    echo "In testing, doubling from 50k to 100k did NOT"
    echo "improve accuracy or recall. It reduced false"
    echo "positives but at the cost of missing more real"
    echo "wake words."
    echo ""
    read -p "Samples [default: $N_SAMPLES]: " INPUT
    N_SAMPLES="${INPUT:-$N_SAMPLES}"
    echo ""

    # --- Augmentation rounds ---
    echo "----------------------------------------------"
    echo "  Augmentation Rounds              default: 2"
    echo "----------------------------------------------"
    echo "How many times each clip is re-processed with"
    echo "different background noise, reverb, and room"
    echo "conditions."
    echo ""
    echo "More rounds = better noise/environment handling."
    echo "Fewer rounds = faster training."
    echo ""
    echo "In testing, 3 rounds produced no measurable"
    echo "improvement over 2 for wake word detection."
    echo "May help if you plan to use this where there's"
    echo "plenty of ambient noise."
    echo ""
    read -p "Augmentation rounds [default: $AUGMENTATION_ROUNDS]: " INPUT
    AUGMENTATION_ROUNDS="${INPUT:-$AUGMENTATION_ROUNDS}"
    echo ""

    # --- Training steps ---
    echo "----------------------------------------------"
    echo "  Training Steps                default: 100000"
    echo "----------------------------------------------"
    echo "How many steps the neural network trains."
    echo "More steps gives the model more time to learn,"
    echo "but with diminishing returns."
    echo ""
    echo "This is the most GPU-intensive phase."
    echo "100k steps on an RTX 4090 takes ~20 minutes."
    echo "150k steps did not improve results in testing."
    echo ""
    read -p "Training steps [default: $TRAINING_STEPS]: " INPUT
    TRAINING_STEPS="${INPUT:-$TRAINING_STEPS}"
    echo ""

    # --- Layer size ---
    echo "----------------------------------------------"
    echo "  Layer Size (Neurons)             default: 32"
    echo "----------------------------------------------"
    echo "The number of neurons in each hidden layer."
    echo "More neurons = more capacity to represent"
    echo "subtle differences in pronunciation."
    echo ""
    echo "In testing, 64 neurons produced statistically"
    echo "identical results to 32 for wake word models."
    echo "The output model stays tiny either way (~200 KB)."
    echo ""
    read -p "Layer size [default: $LAYER_SIZE]: " INPUT
    LAYER_SIZE="${INPUT:-$LAYER_SIZE}"
    echo ""
fi

# Validation samples = 10% of training samples
N_SAMPLES_VAL=$((N_SAMPLES / 10))

# --- Confirm ---
echo "----------------------------------------------"
echo "  Ready to Train"
echo "----------------------------------------------"
echo ""
echo "  Wake word:           \"$WAKE_WORD\""
echo "  Shared memory:       $SHM_SIZE"
echo "  Samples:             $N_SAMPLES"
echo "  Augmentation rounds: $AUGMENTATION_ROUNDS"
echo "  Training steps:      $TRAINING_STEPS"
echo "  Layer size:          $LAYER_SIZE"
echo ""
read -p "Start training? [Y/n] " -r
echo ""
if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    echo "Aborted. Rerun ./train-wakeword.sh when ready."
    exit 0
fi

# =========================================================================
# Launch container
# =========================================================================
echo "Starting training container..."
echo "Container name: $CONTAINER_NAME"
echo ""
echo "=============================================="
echo ""

DOCKER_ARGS=(
    --gpus all
    --shm-size=$SHM_SIZE
    --name "$CONTAINER_NAME"
    --rm
    -v "$OUTPUT_DIR:/output:rw"
    -e WAKE_WORD="$WAKE_WORD"
    -e MODEL_NAME="$MODEL_NAME"
    -e N_SAMPLES="$N_SAMPLES"
    -e N_SAMPLES_VAL="$N_SAMPLES_VAL"
    -e AUGMENTATION_ROUNDS="$AUGMENTATION_ROUNDS"
    -e TRAINING_STEPS="$TRAINING_STEPS"
    -e LAYER_SIZE="$LAYER_SIZE"
)

if [ "$STANDALONE" == "true" ]; then
    DOCKER_ARGS+=( -e STANDALONE=1 )
    DOCKER_ARGS+=( -e TARBALL_URL="$TARBALL_URL" )
else
    DOCKER_ARGS+=( -v "$DATA_DIR:/data:ro" )
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"

echo ""
echo "=============================================="
echo "Training complete!"
echo ""
echo "Model files are in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
echo ""
echo "To use the model:"
echo "  cp $OUTPUT_DIR/${MODEL_NAME}.tflite ~/.local/share/openwakeword/"
echo "=============================================="
