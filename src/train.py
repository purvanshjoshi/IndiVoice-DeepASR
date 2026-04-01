import os
import argparse
import torch
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding
import jiwer

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * jiwer.wer(label_str, pred_str)

    return {"wer": wer}

def train():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA (Kaggle-Optimized)")
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium", help="Base model name")
    parser.add_argument("--train_manifest", type=str, default="data/processed/svarah_manifest.json", help="Path to training manifest")
    parser.add_argument("--val_manifest", type=str, default="data/processed/svarah_manifest.json", help="Path to validation manifest")
    
    # 0. Environment Detection
    is_kaggle = os.path.exists("/kaggle/working")
    default_out = "/kaggle/working/models/whisper-indian-lora" if is_kaggle else "./models/whisper-indian-lora"
    
    parser.add_argument("--output_dir", type=str, default=default_out, help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    
    args = parser.parse_args()

    # 1. Hardware Reporting
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✅ Found {num_gpus} GPU(s). Distributed Data Parallel (DDP) active if launched with 'accelerate'.")
    else:
        print("⚠️ WARNING: GPU NOT DETECTED! Training will be extremely slow.")

    # 1.1 Check Manifests
    for path in [args.train_manifest, args.val_manifest]:
        if not os.path.exists(path):
            print(f"❌ Error: Manifest not found at {path}")
            return

    # 2. Load Processor and Model with Optional Quantization
    print(f"Loading processor and model: {args.model_name}...")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="English", task="transcribe")
    
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config else None
    )

    # 3. Prepare Model for PEFT
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 4. Load Datasets
    print("Loading datasets...")
    
    # Data Integrity Check: Verify at least one file exists
    import json
    with open(args.train_manifest, "r") as f:
        first_line = f.readline()
        if first_line:
            sample = json.loads(first_line)
            if not os.path.exists(sample["audio_filepath"]):
                print(f"\n❌ CRITICAL ERROR: Audio file not found at {sample['audio_filepath']}")
                print("This usually means you only uploaded the manifest but not the audio files.")
                print("Tip: Run Section 1 (Setup) in your Kaggle notebook again; the new 'Auto-Recovery' feature will fix this for you!\n")
                return

    train_dataset = prepare_dataset(args.train_manifest, processor.feature_extractor, processor.tokenizer)
    val_dataset = prepare_dataset(args.val_manifest, processor.feature_extractor, processor.tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. Define Training Arguments (DDP Optimized)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        max_steps=5000,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    # 6. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )

    # 7. Check for Checkpoint
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = None
    if os.path.exists(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"Resuming training from checkpoint: {last_checkpoint}")

    # 8. Start Training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 9. Save final model and processor
    final_output_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_output_path)
    processor.save_pretrained(final_output_path)
    print(f"Training complete. Model saved to {final_output_path}")

if __name__ == "__main__":
    train()

