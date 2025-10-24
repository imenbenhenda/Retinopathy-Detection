# 👁️ Retinopathy Detection using Transfer Learning (ResNet50)

## 🎯 Objective
This project implements a deep learning pipeline to detect diabetic retinopathy from retinal fundus images. It utilizes **Transfer Learning** with a **ResNet50** architecture to classify images as either **Healthy** or **Diseased (Retinopathy)**.

### Key components:
1. **PyTorch Model:** A ResNet50 model fine-tuned on the APTOS 2019 dataset.
2. **Two-Phase Training:** A robust training script (`train.py`) that first trains the classifier head and then fine-tunes the deeper layers.
3. **Reproducibility:** The pipeline ensures reproducible results by using a fixed seed (`set_seed()`) for data splitting and model initialization.
4. **Robust Evaluation:** A dedicated script (`evaluate.py`) to load the best model and generate a final classification report and confusion matrix.

---

## 📁 Dataset
- **Source:** [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- **Task:** The original 5-class problem (scores 0-4) has been simplified into a **binary classification** problem:
    - **Class 0 (Healthy):** Original score 0
    - **Class 1 (Diseased):** Original scores 1, 2, 3, or 4
- **Preprocessing:**
    - Resized to **224×224** pixels (for ResNet50)
    - Pixel normalization (mean/std for ImageNet)
    - Data augmentation applied to the training set (RandomHorizontalFlip, RandomRotation)

---

## 🧱 Model Architecture & Training Strategy
- **Type:** Transfer Learning using a pre-trained **ResNet50** from `torchvision`.
- **Input Shape:** `(224, 224, 3)`
- **Training Strategy:** A two-phase approach to leverage pre-trained weights effectively.

1.  **Phase 1: Head Training (3 Epochs)**
    - **Objective:** To train the new, randomly-initialized final `nn.Linear` classifier head.
    - **Layers:** All ResNet50 backbone layers are **frozen**.
    - **Optimizer:** `Adam` with a learning rate of `1e-3`.

2.  **Phase 2: Fine-Tuning (7 Epochs)**
    - **Objective:** To gently adapt the deeper, pre-trained layers to the specifics of retinal images.
    - **Layers:** The final block (`layer4`) and the `fc` head are **unfrozen** and trained.
    - **Optimizer:** `Adam` with a much lower learning rate of `1e-4`.

- **Loss Function:** `nn.CrossEntropyLoss`
- **Key Feature:** **Early Stopping** is used throughout both phases to monitor `validation_loss`. The script automatically saves the model weights from the single best epoch, preventing overfitting and ensuring the final model is the most generalized one.

---

## 📊 Final Results
After running the full training pipeline, the best model (saved by `EarlyStopping`) was evaluated. The following results are **stable and reproducible** (Seed: 42).

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **97.41%** |
| **Validation Loss** | **0.0783** |
| **Macro Avg F1-score** | **0.9741** |
| **Macro Avg Precision** | **0.9740** |
| **Macro Avg Recall** | **0.9743** |
### A Note on Performance and Project Evolution

Initial project explorations achieved peak, non-reproducible scores as high as **~99.3% accuracy**. A deeper analysis revealed two key methodological issues:

1.  **Non-Reproducibility:** The train/validation data split was random on every run, leading to inconsistent results.
2.  **Overfitting:** The original model was trained for a fixed number of epochs. This process did not prevent the model from continuing to train after its performance on validation data had peaked and begun to degrade.

This version of the project has been **rigorously improved** to fix these issues and ensure **scientific validity**. This was achieved by:

1.  **Implementing `set_seed(42)`:** This ensures the train/validation split is fixed and reproducible.
2.  **Implementing `EarlyStopping`:** The training script now monitors the `validation_loss`. It automatically stops the training process when the model's performance stops improving and saves *only* the model from the best-performing epoch.

The final, reproducible score of **97.41% Accuracy** is therefore the honest and reliable measure of the model's true.

---

## ⚙️ Technologies Used
- Python
- PyTorch
- Pandas & NumPy
- Scikit-learn (for metrics and reporting)
- Matplotlib & Seaborn (for plotting)
- PIL (Pillow)

---

## 📁 Project Structure
```
Retinopathy-Detection/
├── .gitignore # Ignores large files (datasets, models) 
├── data/ # (Not tracked by Git) Dataset folder 
│ ├── train.csv 
│ └── train_images/ 
├── models/ # (Not tracked by Git) Saved models, history, and plots 
│ ├── resnet50_best_checkpoint.pth 
│ ├── training_history.png 
│ └── confusion_matrix.png 
├── notebooks/ # Jupyter notebooks for data exploration 
│ └── exploration.ipynb ├── src/ # Main source code 
│ ├── train.py # Script to train the model 
│ └── evaluate.py # Script to evaluate the best model 
├── requirements.txt # Python dependencies 
└── README.md # Project documentation
```
---

## 👩‍💻 Author
**Imen Ben Henda**
Computer Engineering Student
Focused on AI for healthcare and robust model development.
