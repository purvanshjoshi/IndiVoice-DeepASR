import torch
import json
from datasets import Dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer

def load_manifest(manifest_path):
    """
    Loads a JSONL manifest file.
    """
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_dataset(manifest_path, feature_extractor, tokenizer):
    """
    Creates a Hugging Face Dataset from a manifest and applies preprocessing.
    Uses memory-mapped loading and batched processing for RAM efficiency.
    """
    from datasets import load_dataset
    # Load directly from JSONL using memory-mapping
    dataset = load_dataset("json", data_files=manifest_path, split="train")
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    
    def preprocess_function(batch):
        # Process audio in batches for speed and memory efficiency
        # audio_filepath column returns a list of dictionaries when batched=True
        audio_data = batch["audio_filepath"]
        audio_arrays = [x["array"] for x in audio_data]
        
        # Vectorize audio
        inputs = feature_extractor(audio_arrays, sampling_rate=16000)
        batch["input_features"] = inputs.input_features
        
        # Tokenize text
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    # Configure mapping for Colab stability
    dataset = dataset.map(
        preprocess_function, 
        batched=True,
        batch_size=8, # Small batches to prevent RAM spikes
        remove_columns=dataset.column_names, 
        num_proc=1,
        writer_batch_size=50, # More frequent disk flushes
        desc="Vectorizing datasets (RAM Optimized)"
    )
    return dataset

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch
