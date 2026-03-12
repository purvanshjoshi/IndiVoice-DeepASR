import os
import argparse
import subprocess
from datasets import load_dataset

def download_svarah(base_path):
    print("🚀 Downloading Svarah Dataset from Hugging Face...")
    try:
        # Svarah on HF only has a 'test' split
        dataset = load_dataset("ai4bharat/Svarah", split="test")
        print("✅ Svarah Loaded/Cached from Hugging Face (using 'test' split).")
        
        # Also clone the repo for eval scripts as requested by the user's research docs
        repo_path = os.path.join(base_path, "svarah_repo")
        if not os.path.exists(repo_path):
            subprocess.run(["git", "clone", "https://github.com/AI4Bharat/Svarah.git", repo_path])
            print("✅ Svarah Repo Cloned (for scripts).")
    except Exception as e:
        print(f"❌ Error downloading Svarah: {e}")

def download_indic_accent_db(base_path):
    print("🚀 Downloading IndicAccentDb from Hugging Face...")
    try:
        dataset = load_dataset("DarshanaS/IndicAccentDb")
        print("✅ IndicAccentDb loaded/cached.")
    except Exception as e:
        print(f"❌ Error downloading IndicAccentDb: {e}")

def download_nptel2020(base_path):
    print("⚠️ WARNING: NPTEL2020 is extremely large (15,700 hours, ~500GB+).")
    print("🚀 Cloning NPTEL2020 repository for download scripts...")
    target_path = os.path.join(base_path, "nptel2020_repo")
    if not os.path.exists(target_path):
        subprocess.run(["git", "clone", "https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset.git", target_path])
        print("✅ NPTEL2020 repo cloned. Use scripts inside to download subsets.")
    else:
        print("⏭️ NPTEL2020 repo already exists.")

def download_spire_sies(base_path):
    print("🚀 Information for SPIRE-SIES...")
    print("Reference: https://arxiv.org/abs/2312.00698")
    print("Note: SPIRE-SIES often requires requesting access or following the paper's data release instructions.")

def download_accent_db(base_path):
    print("🚀 Information for Accent DB...")
    print("Link: https://accentdb.org/")
    print("Note: Please download the Indian accent subset manually from the website if automated scripts are unavailable.")

def main():
    parser = argparse.ArgumentParser(description="IndiVoice-DeepASR Data Downloader")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Target directory for raw data")
    parser.add_argument("--dataset", type=str, choices=["svarah", "indic_accent", "nptel", "spire", "accentdb", "all"], default="all")
    
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.dataset in ["svarah", "all"]:
        download_svarah(args.data_dir)
        
    if args.dataset in ["indic_accent", "all"]:
        download_indic_accent_db(args.data_dir)

    if args.dataset in ["nptel", "all"]:
        download_nptel2020(args.data_dir)

    if args.dataset in ["spire", "all"]:
        download_spire_sies(args.data_dir)

    if args.dataset in ["accentdb", "all"]:
        download_accent_db(args.data_dir)

if __name__ == "__main__":
    main()
