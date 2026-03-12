import os
import argparse
import torchaudio
import json
import torch
from tqdm import tqdm
from torchaudio.transforms import Resample
from datasets import load_dataset

def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    Resamples audio to target sampling rate, converts to mono, and saves to output path.
    """
    try:
        waveform, sr = torchaudio.load(input_path)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed audio
        torchaudio.save(output_path, waveform, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def create_manifest(audio_dir, transcript_dir, output_manifest):
    """
    Creates a JSONL manifest file for local files.
    """
    manifest_entries = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    print(f"Creating manifest for {len(audio_files)} files...")
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        transcript_path = os.path.join(transcript_dir, audio_file.replace('.wav', '.txt'))
        
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            manifest_entries.append({
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript
            })
            
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Manifest saved to {output_manifest}")

def process_hf_dataset(dataset_name, output_dir, manifest_path, target_sr=16000):
    """
    Downloads, standardizes, and saves a Hugging Face dataset to local drive.
    Automatically detects available splits (train, test, validation).
    """
    print(f"Loading Hugging Face dataset: {dataset_name}...")
    
    ds = None
    for split_choice in ["train", "test", "validation"]:
        try:
            ds = load_dataset(dataset_name, split=split_choice)
            print(f"✅ Successfully loaded '{split_choice}' split.")
            break
        except Exception:
            continue
            
    if ds is None:
        print(f"❌ Error: Could not load any valid split (train/test/validation) for {dataset_name}. Check permissions or dataset name.")
        return

    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []
    print(f"Preprocessing {len(ds)} samples...")
    
    for i, item in enumerate(tqdm(ds)):
        audio_data = item["audio"]
        # Handle different common text field names
        text = item.get("text") or item.get("sentence") or item.get("transcript") or item.get("transcription")
        
        if audio_data is None or text is None:
            continue
            
        # Standardize
        waveform = torch.tensor(audio_data["array"]).unsqueeze(0)
        sr = audio_data["sampling_rate"]
        
        if sr != target_sr:
            resampler = Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # Save audio
        audio_filename = f"hf_{i}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        torchaudio.save(audio_path, waveform, target_sr)
        
        duration = waveform.shape[1] / target_sr
        manifest_entries.append({
            "audio_filepath": audio_path,
            "duration": duration,
            "text": text
        })

    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"HF Dataset processed and manifest saved to {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="IndiVoice-DeepASR Audio Preprocessing")
    parser.add_argument("--input_dir", type=str, help="Directory containing raw audio")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed audio")
    parser.add_argument("--transcript_dir", type=str, help="Directory containing transcripts")
    parser.add_argument("--manifest_path", type=str, help="Path to save the generated manifest")
    parser.add_argument("--hf_dataset", type=str, help="Hugging Face dataset name (e.g. DarshanaS/IndicAccentDb)")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sampling rate (default: 16000)")
    
    args = parser.parse_args()
    
    if args.hf_dataset:
        process_hf_dataset(args.hf_dataset, args.output_dir, args.manifest_path, args.target_sr)
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist.")
            return

        raw_files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"Standardizing {len(raw_files)} audio files... ")
        
        for audio_file in tqdm(raw_files):
            input_path = os.path.join(args.input_dir, audio_file)
            output_path = os.path.join(args.output_dir, audio_file)
            preprocess_audio(input_path, output_path, args.target_sr)
            
        if args.manifest_path and args.transcript_dir:
            create_manifest(args.output_dir, args.transcript_dir, args.manifest_path)
    else:
        print("Please provide --input_dir or --hf_dataset")

if __name__ == "__main__":
    main()
