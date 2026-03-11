import os
import argparse
import subprocess
from datasets import load_dataset

def download_svarah(base_path):
    print("🚀 Downloading Svarah Dataset...")
    target_path = os.path.join(base_path, "svarah")
    if not os.path.exists(target_path):
        subprocess.run(["git", "clone", "https://github.com/AI4Bharat/Svarah.git", target_path])
        print("✅ Svarah Cloned.")
    else:
        print("⏭️ Svarah already exists.")

def download_indic_accent_db(base_path):
    print("🚀 Downloading IndicAccentDb from Hugging Face...")
    # This just caches it locally for the datasets library to use
    try:
        dataset = load_dataset("DarshanaS/IndicAccentDb")
        print("✅ IndicAccentDb loaded/cached.")
    except Exception as e:
        print(f"❌ Error downloading IndicAccentDb: {e}")

def main():
    parser = argparse.ArgumentParser(description="IndiVoice-DeepASR Data Downloader")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Target directory for raw data")
    parser.add_argument("--dataset", type=str, choices=["svarah", "indic_accent", "all"], default="all")
    
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.dataset in ["svarah", "all"]:
        download_svarah(args.data_dir)
        
    if args.dataset in ["indic_accent", "all"]:
        download_indic_accent_db(args.data_dir)

if __name__ == "__main__":
    main()
