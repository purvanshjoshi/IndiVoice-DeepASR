# Kaggle Migration Guide: IndiVoice-DeepASR

Since we have created a dedicated `kaggle/` workspace, migrating is now much simpler.

## 1. Prepare your Resources (Crucial)
Kaggle needs to see your data from Drive.
1.  **Download from Drive**: Download your `data/processed/` folder (contains `svarah_manifest.json`) and your `models/whisper-indian-lora/checkpoint-100` folder.
2.  **Create Kaggle Dataset**: 
    - Go to Kaggle -> Create -> New Dataset.
    - Upload the `checkpoint-100` folder and the `svarah_manifest.json`.
    - Name the dataset: `indivoice-resumption`.

## 2. Start the Kaggle Notebook
1.  **Create Notebook**: Create a new Python notebook on Kaggle.
2.  **Settings (Right Sidebar)**:
    - **Accelerator**: GPU T4 x2.
    - **Internet**: On.
3.  **Add Data**: Click "+ Add Data" and search for your `indivoice-resumption` dataset.

## 3. The One-Minute Setup
Paste and run this in the first cell of your Kaggle notebook:
```bash
!git clone https://github.com/purvanshjoshi/IndiVoice-DeepASR.git
%cd IndiVoice-DeepASR
!bash kaggle/setup_kaggle.sh
```

## 4. Run Training
The setup script handles the "bridge" between Kaggle's input folder and our repository. Just run the training command:
```bash
!python src/train.py \
    --train_manifest data/processed/svarah_manifest.json \
    --val_manifest data/processed/svarah_manifest.json \
    --output_dir /kaggle/working/whisper-indian-lora
```

## 5. Saving Results
Kaggle's `/kaggle/working` directory is cleared when the session ends. 
*   **Recommendation**: Use the `Save & Run All` button. Once finished, you can download the final model from the "Output" section of the notebook version.
