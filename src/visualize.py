import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_mel_spectrogram(audio_path, output_path=None, title="Mel-Spectrogram", sr=16000):
    """
    Generates and optionally saves a Mel-Spectrogram for a given audio file.
    
    Args:
        audio_path (str): Path to the input audio file.
        output_path (str, optional): Path to save the resulting image. If None, it just returns the figure.
        title (str): Title for the plot.
        sr (int): Target sampling rate.
    """
    # 1. Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 2. Compute Mel-Spectrogram
    # Using parameters that align with Whisper's preprocessing (80-channel Mel)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 3. Create the Visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Spectrogram saved to {output_path}")
        plt.close(fig)
        return output_path
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="IndiVoice Audio Visualization Utility")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="spectrogram.png", help="Path to save output image")
    parser.add_argument("--title", type=str, default="IndiVoice Mel-Spectrogram", help="Title for the plot")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: File {args.input} not found.")
        return
        
    plot_mel_spectrogram(args.input, args.output, args.title)

if __name__ == "__main__":
    main()
