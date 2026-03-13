import os
import argparse
import torchaudio
import json
import torch
from tqdm import tqdm
from torchaudio.transforms import Resample
from datasets import load_dataset, Audio

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
    print(f"--- IndiVoice Preprocessing Engine v1.2 (Latest Fix) ---")
    print(f"Loading Hugging Face dataset: {dataset_name}...")
    
    ds = None
    for split_choice in ["train", "test", "validation"]:
        try:
            ds = load_dataset(dataset_name, split=split_choice)
            print(f"✅ Successfully loaded '{split_choice}' split.")
            break
        except Exception:
            continue
            
    # Identify audio and text columns
    audio_cols = [c for c in ds.column_names if c in ["audio", "audio_filepath", "path", "file"]]
    text_cols = [c for c in ds.column_names if c in ["text", "sentence", "transcript", "transcription", "transcript_clean"]]
    
    if not audio_cols:
        print(f"❌ Error: Could not find an audio column in {dataset_name}. Found: {ds.column_names}")
        return
    if not text_cols:
        print(f"❌ Error: Could not find a text/transcript column in {dataset_name}. Found: {ds.column_names}")
        return
    
    audio_key = audio_cols[0]
    text_key = text_cols[0]
    
    # Force cast the audio column to Audio feature so HF handles decoding
    print(f"Using '{audio_key}' for audio and '{text_key}' for transcripts.")
    ds = ds.cast_column(audio_key, Audio(sampling_rate=target_sr))
    
    # Force python format to ensure dicts are returned when indexed
    ds = ds.with_format("python")
    
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []
    print(f"Preprocessing {len(ds)} samples...")
    
    # Use index-based access to reliably trigger decoders in all environments
    for i in tqdm(range(len(ds))):
        try:
            item = ds[i]
            audio_data = item.get(audio_key)
            text = item.get(text_key)
            
            if not audio_data or not text:
                continue
            
            # Robust extraction of waveform and sampling rate
            waveform = None
            sr = target_sr
            
            # Layer 1: Standard dictionary (already decoded)
            if isinstance(audio_data, dict):
                if "array" in audio_data:
                    waveform = torch.tensor(audio_data["array"]).unsqueeze(0)
                    sr = audio_data.get("sampling_rate", target_sr)
            
            # Layer 2: torchcodec specific handling (CRITICAL for Colab)
            if waveform is None:
                try:
                    # check if it's a torchcodec object or has get_all_samples
                    if hasattr(audio_data, "get_all_samples"):
                        samples = audio_data.get_all_samples()
                        # If it returns a dict with 'data' and 'sample_rate'
                        if hasattr(samples, "data"):
                            waveform = samples.data
                            if waveform.ndim == 1:
                                waveform = waveform.unsqueeze(0)
                            sr = getattr(samples, "sample_rate", target_sr)
                        else:
                            waveform = torch.tensor(samples).unsqueeze(0)
                    elif hasattr(audio_data, "decode"):
                        decoded = audio_data.decode()
                        if isinstance(decoded, dict) and "array" in decoded:
                            waveform = torch.tensor(decoded["array"]).unsqueeze(0)
                            sr = decoded.get("sampling_rate", target_sr)
                except Exception as e:
                    # Silently fail layer 2 as usual
                    pass

            # Layer 3: Callable decoder object (common in some HF versions)
            if waveform is None and callable(audio_data):
                try:
                    decoded = audio_data()
                    if isinstance(decoded, dict) and "array" in decoded:
                        waveform = torch.tensor(decoded["array"]).unsqueeze(0)
                        sr = decoded.get("sampling_rate", target_sr)
                    elif hasattr(decoded, "array"):
                         waveform = torch.tensor(getattr(decoded, "array")).unsqueeze(0)
                         sr = getattr(decoded, "sampling_rate", target_sr)
                except Exception:
                    pass
            
            # Layer 4: Object with 'array' or 'sampling_rate' attributes directly
            if waveform is None:
                try:
                    if hasattr(audio_data, "array"):
                        waveform = torch.tensor(getattr(audio_data, "array")).unsqueeze(0)
                        sr = getattr(audio_data, "sampling_rate", target_sr)
                except Exception:
                    pass

            if waveform is None:
                # Last resort: Try to cast to numpy array or torch tensor directly
                try:
                    if torch.is_tensor(audio_data):
                        waveform = audio_data.unsqueeze(0) if audio_data.ndim == 1 else audio_data
                    else:
                        import numpy as np
                        arr = np.array(audio_data)
                        waveform = torch.from_numpy(arr).unsqueeze(0)
                except Exception:
                    print(f"Skipping {i}: Could not decode audio_data of type {type(audio_data)}")
                    # Print attributes to help debug if it fails again
                    print(f"Debug Info - Sample {i} attributes: {dir(audio_data)}")
                    continue
        except Exception as e:
            print(f"Skipping {i} due to unhandled error: {e}")
            continue

        # Standardize to target sample rate (should already be done by cast_column)
        if sr != target_sr:
            resampler = Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # Standardize to Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Save audio locally in Move to processed
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
