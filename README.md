<div align="center">
  <img src="assets/banner.png" alt="IndiVoice Banner" width="100%">
  
  # 🎧 IndiVoice-DeepASR: Indian-Accented Speech Recognition
  
  **Bridging the Accent Gap in Modern ASR with Whisper + LoRA**
  
  [![GitHub Stars](https://img.shields.io/github/stars/purvanshjoshi/IndiVoice-DeepASR?style=for-the-badge&logo=github&color=FFD700)](https://github.com/purvanshjoshi/IndiVoice-DeepASR/stargazers)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Svarah-blue?style=for-the-badge)](https://huggingface.co/datasets/ai4bharat/Svarah)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge)](LICENSE)

  [**Explore the Code**](https://github.com/purvanshjoshi/IndiVoice-DeepASR) • [**Launch Colab**](https://colab.research.google.com/github/purvanshjoshi/IndiVoice-DeepASR/blob/main/notebooks/IndiVoice_Colab_Entry.ipynb) • [**Launch Kaggle**](https://github.com/purvanshjoshi/IndiVoice-DeepASR/tree/main/kaggle)

  > [!IMPORTANT]
  > **Ultra-Resilience Update (v1.8)**: Added high-frequency checkpointing (every 100 steps) and auto-resumption to protect against Colab/Kaggle runtime disconnections. Resolved `load_best_model_at_end` compatibility issues.
</div>

---

## 🌟 Overview

Current commercial ASR systems suffer from a **20-30% performance drop** when processing Indian English accents. **IndiVoice-DeepASR** is a research-driven project that fine-tunes OpenAI's Whisper models using **LoRA (Low-Rank Adaptation)** to achieve state-of-the-art accuracy across diverse Indian linguistic profiles.

### ✨ Key Features
- **🛡️ Ultra-Resilient**: Automatic checkpoint detection and resumption. Never lose more than 10-15 minutes of training.
- **🚀 Efficiency**: Fine-tune with < 2% of total parameters using PEFT techniques.
- **🇮🇳 Localization**: Optimized for Hindi, Tamil, Kannada, Bengali, and Punjabi accents.
- **🌊 Stable Decoding**: Multi-layered `AudioDecoder` logic for robust preprocessing on diverse system environments.
- **⚡ Performance**: Achieve significant WER reduction compared to base Whisper models.

---

## 🛠️ Tech Stack & Pillars

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Model Backbone</b><br><img src="https://img.shields.io/badge/Transformers-8A2BE2?style=flat-square&logo=huggingface&logoColor=white" alt="HF"></td>
      <td align="center"><b>Optimization</b><br><img src="https://img.shields.io/badge/PEFT/LoRA-000000?style=flat-square&logo=github&logoColor=white" alt="PEFT"></td>
      <td align="center"><b>Audio Engine</b><br><img src="https://img.shields.io/badge/Torchaudio-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="Audio"></td>
    </tr>
    <tr>
      <td align="center"><b>Cloud Compute</b><br><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=Kaggle&logoColor=white" alt="Kaggle"> <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Colab"></td>
      <td align="center"><b>Deployment</b><br><img src="https://img.shields.io/badge/Gradio-FF9D00?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"></td>
      <td align="center"><b>Infrastructure</b><br><img src="https://img.shields.io/badge/NVIDIA_CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA"></td>
    </tr>
  </table>
</div>

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Raw Audio] --> B(Standardization: 16kHz Mono)
    B --> C{IndiVoice Engine}
    C --> D[Whisper Backbone]
    C --> E[LoRA Adapters]
    D & E --> F[Optimized Transcripts]
    F --> G[Metric Analysis: WER/CER]
```

---

## 🚀 Quick Start

### 1. Collaborative Training (Recommended)
Choose your preferred platform for free GPU access:
- [**Colab Gateway**](https://colab.research.google.com/github/purvanshjoshi/IndiVoice-DeepASR/blob/main/notebooks/IndiVoice_Colab_Entry.ipynb): Best for initial setup and rapid experimentation.
- [**Kaggle Runner**](https://github.com/purvanshjoshi/IndiVoice-DeepASR/tree/main/kaggle): Best for long-running training (30 hours/week free GPU). Includes a specialized `setup_kaggle.sh` for one-click environment configuration.

### 2. Local Development
```bash
# Clone & Install
git clone https://github.com/purvanshjoshi/IndiVoice-DeepASR.git
cd IndiVoice-DeepASR
pip install -r requirements.txt

# Preprocess (Multi-layered decoder support)
python src/preprocess.py --hf_dataset ai4bharat/Svarah --output_dir data/processed

# Train (Auto-resumes from latest checkpoint)
python src/train.py --output_dir models/indian-accent-lora
```

---

## 📂 Repository Structure

```text
IndiVoice-DeepASR/
├── assets/            # Branding & Visuals
├── kaggle/            # Dedicated Kaggle training workspace
├── src/               # Optimized Pipeline Scripts (Train/Preprocess/Deploy)
├── notebooks/         # Interactive Research
├── data/              # Dataset Symlinks & Manifests
├── models/            # Checkpoints & LoRA Weights
└── paper/             # ICASSP Publication Source
```

---

## 🎓 Academic Citation

If you use this work in your research, please cite:

```bibtex
@misc{indivoice2026,
  author = {Purvansh Joshi and Archit Mittal},
  title = {IndiVoice-DeepASR: Efficient Adaptation of Multilingual Speech Models for Indian Accents},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/purvanshjoshi/IndiVoice-DeepASR}}
}
```

---

<div align="center">
  <p>Built with ❤️ for the Indian Speech Recognition Research Community</p>
  <img src="https://img.shields.io/badge/Made%20with-Python-blue" alt="Python">
</div>
