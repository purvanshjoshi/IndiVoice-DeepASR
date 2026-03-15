#!/bin/bash
# IndiVoice-DeepASR: Kaggle Setup Script
# Use this to prepare the Kaggle environment for training.

echo "🚀 Starting Kaggle Environment Setup..."

# 1. Path Configuration
KAGGLE_INPUT="/kaggle/input"
KAGGLE_WORKING="/kaggle/working/IndiVoice-DeepASR"
REPO_URL="https://github.com/purvanshjoshi/IndiVoice-DeepASR.git"

# 2. Clone Repository if not present
if [ ! -d "$KAGGLE_WORKING" ]; then
    echo "📦 Cloning repository..."
    mkdir -p /kaggle/working
    cd /kaggle/working
    git clone $REPO_URL
    cd IndiVoice-DeepASR
else
    echo "✅ Repository already exists at $KAGGLE_WORKING"
    cd "$KAGGLE_WORKING"
    git fetch --all
    git reset --hard origin/main
fi

# 3. Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/processed
mkdir -p models/whisper-indian-lora

# 4. Link Kaggle Input Datasets/Checkpoints
# Note: User must add the dataset to the Kaggle notebook first.
# Example: Adding 'indivoice-processed-data' dataset
if [ -d "$KAGGLE_INPUT/indivoice-processed-data" ]; then
    echo "🔗 Linking processed data from Kaggle Input..."
    ln -s "$KAGGLE_INPUT/indivoice-processed-data/svarah_manifest.json" data/processed/svarah_manifest.json
fi

# Example: Adding 'indivoice-checkpoints' dataset
if [ -d "$KAGGLE_INPUT/indivoice-checkpoints" ]; then
    echo "🔗 Copying previous checkpoint for resumption..."
    cp -r $KAGGLE_INPUT/indivoice-checkpoints/checkpoint-* models/whisper-indian-lora/
fi

# 5. Install Dependencies
echo "🛠️ Installing dependencies (this may take a minute)..."
pip install -r requirements.txt --quiet

echo "✨ Setup Complete! You can now run the training script."
