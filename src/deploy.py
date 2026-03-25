import os
import argparse
import gradio as gr
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
from visualize import plot_mel_spectrogram

def load_indivoice_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path {model_path} does not exist.")
        print("Please train the model before launching the demo.")
        import sys
        sys.exit(1)
    print(f"Loading IndiVoice-DeepASR model from {model_path}...")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Try to load processor from local path first, else fallback to base model
    try:
        print("Loading processor from local path...")
        processor = WhisperProcessor.from_pretrained(model_path)
    except Exception:
        print("Local processor not found, loading from base model...")
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor

def transcribe(audio_path, model, processor):
    if audio_path is None:
        return "Please upload or record an audio file.", None
    
    # 1. Generate Spectrogram
    import tempfile
    temp_spec = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plot_mel_spectrogram(audio_path, output_path=temp_spec, title="IndiVoice: Acoustic Fingerprint")
    
    # 2. Process Audio for Inference
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        
    input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    
    # 3. Generate Transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription, temp_spec

def launch_demo():
    parser = argparse.ArgumentParser(description="Launch IndiVoice-DeepASR Gradio Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--share", action="store_true", help="Create a public URL")
    
    args = parser.parse_args()

    model, processor = load_indivoice_model(args.model_path)

    interface = gr.Interface(
        fn=lambda audio: transcribe(audio, model, processor),
        inputs=gr.Audio(type="filepath", label="Upload or Record Indian English Audio"),
        outputs=[
            gr.Textbox(label="IndiVoice Transcription"),
            gr.Image(label="Mel-Spectrogram (Whisper Input Feature)")
        ],
        title="🎧 IndiVoice-DeepASR: Acoustic Intelligence Demo",
        description="Fine-tuned Whisper for Indian Accents. View the Mel-spectrogram to see exactly what the AI 'hears'.",
        theme="huggingface"
    )

    interface.launch(share=args.share)

if __name__ == "__main__":
    launch_demo()

