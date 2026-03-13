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
    """
    data = load_manifest(manifest_path)
    
    # Convert to HF Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    
    def preprocess_function(batch):
        # Process audio
        audio = batch["audio_filepath"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        
        # Process text
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(
        preprocess_function, 
        remove_columns=dataset.column_names, 
        num_proc=1,
        writer_batch_size=100, # Periodically flush to disk to save RAM
        desc="Vectorizing datasets"
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
