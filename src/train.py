import os
import argparse
import torch
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
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
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA")
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium", help="Base model name")
    parser.add_argument("--train_manifest", type=str, default="data/processed/train_manifest.json", help="Path to training manifest")
    parser.add_argument("--val_manifest", type=str, default="data/processed/val_manifest.json", help="Path to validation manifest")
    parser.add_argument("--output_dir", type=str, default="./models/whisper-indian-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size (lowered for VRAM)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()

    # 0. Check Environment
    if not torch.cuda.is_available():
        print("\n" + "="*50)
        print("⚠️ WARNING: GPU NOT DETECTED!")
        print("Training Whisper on CPU will be 50-100x slower.")
        print("If you are in Colab, go to: Runtime -> Change runtime type -> T4 GPU.")
        print("="*50 + "\n")

    # 0.1 Check Manifests
    for path in [args.train_manifest, args.val_manifest]:
        if not os.path.exists(path):
            print(f"❌ Error: Manifest not found at {path}")
            print("Please run Section 4 (Preprocessing) first to generate this file.")
            return

    # 1. Load Processor and Model
    print(f"Loading processor and model: {args.model_name}...")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # 2. Prepare Model for PEFT
    model.config.use_cache = False # Required for gradient checkpointing
    # Necessary for gradient checkpointing in PEFT:
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

    # 3. Load Datasets
    print("Loading datasets...")
    train_dataset = prepare_dataset(args.train_manifest, processor.feature_extractor, processor.tokenizer)
    val_dataset = prepare_dataset(args.val_manifest, processor.feature_extractor, processor.tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, # Maintain effective batch size
        learning_rate=args.learning_rate,
        warmup_steps=50,
        max_steps=5000, # Adjust based on dataset size
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=4, # Lowered from 8
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
        gradient_checkpointing=True, # Critical for fitting Whisper on 15GB VRAM
    )

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )

    # 6. Check for Checkpoint
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = None
    if os.path.exists(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            print("No checkpoint found. Starting training from scratch...")

    # 7. Start Training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 8. Save final model and processor
    final_output_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_output_path)
    processor.save_pretrained(final_output_path)
    print(f"Training complete. Model and processor saved to {final_output_path}")

if __name__ == "__main__":
    train()

