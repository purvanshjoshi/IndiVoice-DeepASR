---
description: How to migrate training from Colab to Kaggle
---

# 🚢 Workflow: Kaggle Migration

Follow these steps to move your workload to Kaggle for long-running GPU sessions.

1.  **Push Latest Code**: Ensure GitHub has your latest scripts.
    ```bash
    git add .
    git commit -m "Sync before migration"
    git push origin master
    ```

2.  **Download Data**: Download `data/processed/svarah_manifest.json` and your latest `checkpoint-XXX` from Google Drive.

3.  **Kaggle Setup**:
    - Create a Kaggle Dataset named `indivoice-resumption` with the downloaded files.
    - Create a new Kaggle Notebook with **Internet ON** and **GPU T4 x2**.

// turbo
4.  **Initialize Kaggle**: Run the setup script in the first cell.
    ```bash
    !git clone https://github.com/purvanshjoshi/IndiVoice-DeepASR.git
    %cd IndiVoice-DeepASR
    !bash kaggle/setup_kaggle.sh
    ```

5.  **Start Training**: Execute the training command as described in `kaggle/IndiVoice_Kaggle_Trainer.ipynb`.
