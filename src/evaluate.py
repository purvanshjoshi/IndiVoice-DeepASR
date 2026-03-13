import os
import argparse
import torch
import json
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
from utils import load_manifest
import jiwer

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--test_manifest", type=str, default="data/processed/test_manifest.json", help="Path to test manifest")
    parser.add_argument("--output_file", type=str, default="results/test_evaluation.json", help="Path to save results")
    
    args = parser.parse_args()

    # 1. Load Config and Model
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path {args.model_path} does not exist.")
        print("Please run Section 5 (Training) first to train and save the model.")
        return
        
    print(f"Loading model from {args.model_path}...")
    peft_config = PeftConfig.from_pretrained(args.model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Try to load processor from local path first, else fallback to base model
    try:
        print("Loading processor from local path...")
        processor = WhisperProcessor.from_pretrained(args.model_path)
    except Exception:
        print("Local processor not found, loading from base model...")
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

    # 2. Load Test Data
    test_data = load_manifest(args.test_manifest)
    print(f"Evaluating on {len(test_data)} samples...")

    results = []
    total_wer = 0
    total_cer = 0

    # 3. Inference Loop
    for item in tqdm(test_data):
        audio_path = item["audio_filepath"]
        reference = item["text"]
        
        # Load and process audio
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
        
        # Generate Transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
        # Compute Metrics
        current_wer = jiwer.wer(reference, transcription)
        current_cer = jiwer.cer(reference, transcription)
        
        total_wer += current_wer
        total_cer += current_cer
        
        results.append({
            "audio": audio_path,
            "reference": reference,
            "hypothesis": transcription,
            "wer": current_wer,
            "cer": current_cer
        })

    # 4. Final Summary
    avg_wer = (total_wer / len(test_data)) * 100
    avg_cer = (total_cer / len(test_data)) * 100
    
    summary = {
        "average_wer": avg_wer,
        "average_cer": avg_cer,
        "total_samples": len(test_data),
        "details": results
    }

    print(f"\nEvaluation Results:")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")

    # 5. Save Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    evaluate()

