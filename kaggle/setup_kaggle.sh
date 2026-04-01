#!/bin/bash
# IndiVoice-DeepASR: Kaggle Setup Script
# Use this to prepare the Kaggle environment for training.

echo "🚀 Starting Kaggle Environment Setup..."

# 1. Path Configuration
KAGGLE_INPUT="/kaggle/input"
REPO_DIR="/kaggle/working/IndiVoice-DeepASR"
REPO_URL="https://github.com/purvanshjoshi/IndiVoice-DeepASR.git"

# 2. Setup Working Directory
echo "📦 Setting up repository..."
mkdir -p /kaggle/working
cd /kaggle/working

if [ ! -d "IndiVoice-DeepASR" ]; then
    git clone $REPO_URL
else
    cd IndiVoice-DeepASR
    git fetch --all
    git reset --hard origin/master
fi
cd "$REPO_DIR"

# 3. Create Infrastructure
echo "📁 Creating directory structure..."
mkdir -p data/processed
mkdir -p models/whisper-indian-lora

# 4. Link Kaggle Input Datasets/Checkpoints
# Smart detection of 'indivoice-resumption' or other provided datasets
echo "🔗 Searching for input data..."
find $KAGGLE_INPUT -name "svarah_manifest.json" -exec ln -sf {} data/processed/svarah_manifest.json \;

# Resumption Check: If the user provided a 'checkpoint-*' folder in any dataset, copy it
CHECKPOINT_SOURCE=$(find $KAGGLE_INPUT -name "checkpoint-*" -type d | head -n 1)
if [ ! -z "$CHECKPOINT_SOURCE" ]; then
    echo "♻️ Found checkpoint at $CHECKPOINT_SOURCE. Preparing for resumption..."
    cp -rn $CHECKPOINT_SOURCE models/whisper-indian-lora/
fi

# 5. Automated Accelerate Config (Dual-T4 Optimized)
echo "⚙️ Configuring Multi-GPU Accelerator..."
mkdir -p ~/.cache/huggingface/accelerate
cat <<EOF > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# 6. Install Dependencies
echo "🛠️ Installing optimized dependencies..."
pip install -r requirements.txt --quiet
pip install bitsandbytes --quiet # Ensure quantization is available

echo "✨ Kaggle Setup Complete! Repository is ready for Dual-T4 training."
echo "👉 Launch command: accelerate launch src/train.py"
