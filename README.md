# Retinopathy Detection from Retinal Images

## Objective
Develop an automated system for detecting retinopathy from retinal images using Transfer Learning with pre-trained models (ResNet50) for binary classification:
- 0: Healthy
- 1: Retinopathy

## Dataset
[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
- Retinal images for diabetic retinopathy detection
- Binary classification: healthy vs retinopathy
- Medical-grade retinal fundus images

## Results
| Metric | Value |
|--------|-------|
| Validation Loss | ~0.0248 |
| Accuracy | 99.32% |
| F1-Score | ~0.993 |

### Confusion Matrix
Actual \ Predicted 0 (Healthy) 1 (Retinopathy)
0 (Healthy) 369 2
1 (Retinopathy) 3 358

## Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Transfer Learning Approach**:
  - Phase 1: Training only the final classification head
  - Phase 2: Fine-tuning the last layers of ResNet50
- **Output**: Binary classification (Sigmoid activation)

## Training Strategy
- **Optimizer**: Adam with customized learning rates
- **Loss Function**: Binary Cross-Entropy
- **Callbacks**: Early Stopping, Model Checkpointing
- **Data Augmentation**: Rotation, flipping, brightness adjustment

## Key Features
- 👁️ **Medical Imaging** - Retinal fundus image analysis
- 🔄 **Transfer Learning** - Leveraging ResNet50 pre-trained weights
- 📊 **High Performance** - 99.32% accuracy in retinopathy detection
- 🏥 **Clinical Relevance** - Automated screening for diabetic retinopathy

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/your-username/retinopathy-detection.git
cd retinopathy-detection

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py

# Evaluate model
python src/evaluate.py
Project Structure
retinopathy-detection/
├── src/
│   ├── train.py          # Training script (2-phase approach)
│   └── evaluate.py       # Model evaluation and metrics
├── models/               # Trained models and checkpoints
├── data/                 # Dataset and processed images
├── notebooks/            # Data exploration and analysis
└── results/              # Performance metrics and visualizations
Author
Imen Ben Henda - Computer Engineering Student
