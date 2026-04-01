# 🚀 IndiVoice-DeepASR: Kaggle Dual-T4 Master Guide

This guide ensures you extract **100% performance** from Kaggle's dual T4 GPUs. Follow these steps to migrate your training from Colab to a professional, automated workspace.

---

## 1. Prepare your Resources
Kaggle needs to see your data from Drive and GitHub.
1.  **Download from Drive**: Download your `data/processed/` folder (contains `svarah_manifest.json`) and your latest `checkpoint-XXX`.
2.  **Create Kaggle Dataset**: 
    - Go to Kaggle -> Create -> New Dataset.
    - Upload the checkpoint folder and the manifest.
    - Name the dataset: `indivoice-resumption`.
3.  **GH Integration (Optional)**: In your Kaggle account settings, link your GitHub profile to enable direct synchronization.

## 2. Start the Optimized Notebook
1.  **Create Notebook**: Create a new Python notebook on Kaggle.
2.  **Accelerator (CRITICAL)**: In the right sidebar, select **GPU T4 x2**.
3.  **Internet**: Turn **On**.
4.  **Add Data**: Click "+ Add Data" and add your `indivoice-resumption` dataset.

## 3. The One-Minute "Pro" Setup
Paste and run this in the first cell of your Kaggle notebook:
```bash
!git clone https://github.com/purvanshjoshi/IndiVoice-DeepASR.git
%cd IndiVoice-DeepASR
!bash kaggle/setup_kaggle.sh
```
*The setup script will now automatically configure your Dual-GPU environment and link your manifest.*

## 4. Acoustic Intelligence (Pre-flight Check) (NEW)
Check your data quality before training. Run this in a new cell:
```python
from IPython.display import Image, display
from src.visualize import plot_mel_spectrogram
import json

with open("data/processed/svarah_manifest.json", "r") as f:
    sample = json.loads(f.readline())
    audio_path = sample["audio"]

plot_mel_spectrogram(audio_path, output_path="sample_spec.png")
display(Image("sample_spec.png"))
```

## 5. Launch Multi-GPU Training
Use the `accelerate` command to distribute the workload across both T4 GPUs:
```bash
!accelerate launch src/train.py \
    --train_manifest data/processed/svarah_manifest.json \
    --val_manifest data/processed/svarah_manifest.json
```
*Note: This command will automatically save checkpoints to `/kaggle/working/models`.*

## 6. Persistence & Deployment
Kaggle sessions are temporary. To save your hard-earned work:
1.  **Background Training**: Use **'Save Version' -> 'Save & Run All'**. 
2.  **Result Sync**: Link your notebook to GitHub (File -> Link to GitHub).
3.  **Final Model**: Once training finishes, your LoRA adapters will be in the "Output" section. Download the `final` folder for deployment in `src/deploy.py`.

---
> [!TIP]
> **Weekly Limit**: You have 30 hours of free GPU time per week on Kaggle. Use it wisely for long runs!
