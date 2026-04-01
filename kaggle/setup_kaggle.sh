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

# Robust Nesting Protection: While we are inside a directory named IndiVoice-DeepASR, move up
while [[ $(basename $(pwd)) == "IndiVoice-DeepASR" ]]; do
    cd ..
done

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
pip install bitsandbytes --quiet 

# 7. Auto-Recovery: Download Audio if missing
# Trigger if folder missing OR folder is empty (previous failed attempt)
if [[ -f "data/processed/svarah_manifest.json" ]]; then
    SHOULD_RECOVER=false
    if [ ! -d "data/processed/svarah" ]; then
        SHOULD_RECOVER=true
    elif [ -z "$(ls -A data/processed/svarah 2>/dev/null)" ]; then
        SHOULD_RECOVER=true
        rm -rf data/processed/svarah # Remove empty folder to start fresh
    fi

    if [ "$SHOULD_RECOVER" = true ]; then
        echo "⚠️ Audio files missing for Svarah dataset! Launching Auto-Recovery..."
        
        # Check for Hugging Face Token (Gated Dataset Requirement)
        if [[ -z "$HF_TOKEN" ]]; then
            echo "❌ WARNING: HF_TOKEN not found in environment."
            echo "   Svarah is a GATED dataset. To fix this:"
            echo "   1. Add 'HF_TOKEN' to your Kaggle Secrets (Add-ons -> Secrets)."
            echo "   2. Accept the terms at: https://huggingface.co/datasets/ai4bharat/Svarah"
            echo "   Continuing attempt anyway..."
        fi

        mkdir -p data/processed/svarah
        python src/preprocess.py \
            --hf_dataset ai4bharat/Svarah \
            --output_dir data/processed/svarah \
            --manifest_path data/processed/svarah_manifest.json \
            --target_sr 16000
        
        if [ ! -d "data/processed/svarah" ] || [ -z "$(ls -A data/processed/svarah)" ]; then
            echo "❌ Auto-Recovery failed! Please ensure HF_TOKEN is correctly set."
        else
            echo "✅ Auto-Recovery Complete! Audio files downloaded."
        fi
    fi
fi

echo "✨ Kaggle Setup Complete! Repository is ready for Dual-T4 training."
echo "👉 Launch command: accelerate launch src/train.py"
