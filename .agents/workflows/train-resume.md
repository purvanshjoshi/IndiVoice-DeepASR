---
description: How to resume whisper training from the last saved checkpoint
---

# 🚀 Workflow: Resume Training

Follow these steps to continue training your model if the session was interrupted.

1.  **Sync Repository**: Ensure you have the latest code.
    ```bash
    git pull origin master
    ```

2.  **Verify Checkpoint**: Check if a checkpoint exists in your models directory.
    ```bash
    ls -d models/whisper-indian-lora/checkpoint-*
    ```

3.  **Run Training**: The training script is configured to auto-detect the latest checkpoint.
    ```bash
    python src/train.py \
        --train_manifest data/processed/svarah_manifest.json \
        --val_manifest data/processed/svarah_manifest.json \
        --output_dir ./models/whisper-indian-lora
    ```

4.  **Monitor Logs**: Watch the `logging_steps` output to ensure loss is decreasing.
